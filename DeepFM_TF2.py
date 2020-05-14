#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter,deque


# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[3]:


# floatx默认是float32的，np里的都是64，直接都搞成64省事
tf.keras.backend.set_floatx('float64')


# # Data

# load tfrecords
# ```
# advertiser_id, campaign_id, group_id, creative_id, creative_tag, ad_style, bid_price, display_url, marketing_type, province_id, city_id
# ```

# ## tfrecord

# In[ ]:


base_dir="/home/zhoutong/data/apus_ad/hc"
app2hash_fp=base_dir+"/ad_hc_2020-04-22_app2hash.txt"
info_fp = base_dir+"/ad_hc_2020-04-22_info.json"
# trd_gz_fp = base_dir +"/ad_hc_2020-04-22.trd.gz"
trd_gz_fp_list = [base_dir + i for i in ["/ad_hc_2020-04-21.trd.gz","/ad_hc_2020-04-22.trd.gz"]]


# In[ ]:


with open(info_fp,"r") as fr:
    info=json.load(fr)
with open(app2hash_fp,"r") as fr:
    hash2app = dict([i.strip("\n").split("\t")[::-1] for i in fr.readlines()])


# In[ ]:


print(info['count'])
info['detail'].keys()
str_fields=info['detail']['string']
double_fields=info['detail']['double']
arr_int_fields=info['detail']['array<int>']
long_fields=info['detail']['bigint']  # 时间戳 暂时不用，应该在spark里做一下解析
int_fields=info['detail']['int']


# In[ ]:


# one_hot_fields=["device_id_s"]
# multi_hot_fields=["class_id_topN"]
# dense_fields=["download_num_mid"]
feature_desc = {}
# onehot
for f in str_fields:
    feature_desc.update({f:tf.io.FixedLenFeature([], tf.string, default_value='')})
# multi_hot
for f in arr_int_fields:
    feature_desc.update({f:tf.io.VarLenFeature(dtype=tf.int64)})
# dense / numeric
for f in double_fields:
    feature_desc.update({f:tf.io.FixedLenFeature([],dtype=tf.float32,default_value=0)})
for f in int_fields:
    feature_desc.update({f:tf.io.FixedLenFeature([],dtype=tf.int64,default_value=0)})

def _parse_func(inp):
    return tf.io.parse_single_example(inp,feature_desc)



trd=tf.data.TFRecordDataset(trd_gz_fp_list,compression_type='GZIP').map(_parse_func)
for i in trd.take(2):
    print(i)
dataset=trd.shuffle(5*128).batch(128,drop_remainder=True)


# ## Criteo 60k

# In[5]:


base_dir="/home/zhoutong/notebook_collection/tmp/CTR"
data_fp=os.path.join(base_dir,"criteo_data_sampled_60k.csv")

print("[data_fp]: "+data_fp)


# ### 数据观察

# 部分数据观察

# In[6]:


# 正负样本比: 0.254237288=1.5w/(4.4w+1.5w)
df_head10 = pd.read_csv(data_fp, nrows=10)
df_head10.head()
num_feat=[i for i in df_head10.columns if i.startswith("I")]
cat_feat=[i for i in df_head10.columns if i.startswith("C")]
print("[num_feat]: "+ ",".join(num_feat))
print("[cat_feat]: "+ ",".join(cat_feat))
df_head10.groupby("label").agg({"label":"count"})


# 全量数据观察

# In[7]:


df_total=pd.read_csv(data_fp, nrows=600000)
df_total[num_feat].describe()
df_total[num_feat].count()
df_total[cat_feat].describe()
print("所有category特征合计: ",df_total[cat_feat].describe().loc['unique',:].sum())
print(">>> 注意以下的特征unique计数，是排除了NA的")
df_total[cat_feat].describe().loc['unique',:].T
df_total.groupby("label").agg({"label":"count"})


# ### 获取featureMap
# 全量数据60w太大，直接做get_dummies可以用sparse=True
# - 一来偶尔会跑不出来，推测是压力太大速度太慢（跑太久了手动停了）；
# - 二来后续输入tf中，目前已知的只有.values的方案，这无异于把整个df都重新dense化了；
# 
# 这里用自定义的方案，按chunksize过一遍全体数据，拿到featuremap后，手动做sparseTensor

# In[ ]:


# 全量直接做get_dummies容易卡死
df_sparse=pd.get_dummies(df_total,sparse=True)
df_sparse
df_sparse.dtypes
# "{:.4f} mb".format(df_sparse.memory_usage().sum()/1e6)


# In[ ]:





# In[6]:


# 自行遍历构造featureMap
chunksize=1000
df_total_iter=pd.read_csv(data_fp, chunksize=chunksize)
num_feat=[]
cat_feat=[]
cat_featureMap={}
multihot_feat=["C27","C28"]
for idx,chunk in tqdm(enumerate(df_total_iter),total=60*10000/chunksize):
    ###############################
    # 随机构造两个multihot特征
    ###############################
    m1 = []
    for i in range(chunk.shape[0]):
        # 随机 0~100 长度的list，list内的元素是 0~2000 注意str化
        random_size = np.random.randint(low=0,high=100+1)
        m1.append(np.random.randint(low=0,high=2000+1,size=random_size).astype(str))
    chunk['C27'] = m1
    m2 = []
    for i in range(chunk.shape[0]):
        # 随机 0~20 长度的list，list内的元素是 0~100 注意str化
        random_size = np.random.randint(low=0,high=20+1)
        m2.append(np.random.randint(low=0,high=100+1,size=random_size).astype(str))
    chunk['C28'] = m2

    ###############################
    # 提取numeric特征和category特征
    ###############################
    _num_feat=[i for i in chunk.columns if i.startswith("I")]
    _cat_feat=[i for i in chunk.columns if i.startswith("C")]
    if len(num_feat) > 0:
        assert num_feat == _num_feat,f"I系特征不符，前{idx*chunksize}条为: {num_feat}，此后为{_num_feat}"
    else:
        num_feat = _num_feat
    if len(cat_feat) > 0:
        assert cat_feat == _cat_feat,f"C系特征不符，前{idx*chunksize}条为: {cat_feat}，此后为{_cat_feat}"
    else:
        cat_feat = _cat_feat
    
    #############################
    # category特征构造出featureMap
    #############################
    for feat in cat_feat:
#         features=list(chunk[feat].unique())
        features=list(np.unique(np.hstack(chunk[feat].values.flat)))
        features_ori=cat_featureMap.get(feat,[])
        cat_featureMap.update({feat: list(set(features+features_ori))})
    
for k,v in cat_featureMap.items():
    print(f"k:{k}, cnt:{len(v)}")

cat_featureSize=sum([len(v) for k,v in cat_featureMap.items()])
print("所有的category特征总计: {}".format(cat_featureSize))

before=0
cat_featureIdx_beginAt={}
for k,v in cat_featureMap.items():
    cat_featureIdx_beginAt.update({k:before})
    before += len(v)
print("各category特征的起始索引:")
cat_featureIdx_beginAt


# ### 特征工程

# In[7]:


chunk.head(5)


# In[8]:


def normalize(df_inp,num_fields):
    """
    连续特征归一化
    """
    df=df_inp.copy()
    max_records=df[num_fields].apply(np.max,axis=0)
    min_records=df[num_fields].apply(np.min,axis=0)
    denominator=(max_records-min_records).apply(lambda x: x if x!=0 else 1e-4) 
    for f in num_fields:
        df[f] = df[f].apply(lambda x: np.abs(x-min_records[f])/denominator[f])
    return df


def fill_numeric_NA(df_inp,num_fields):
    """
    连续特征的NA填充 | 暂时直接用均值填充
    """
    df=df_inp.copy()
    df_numeric_part=df[num_fields]
    df[num_fields]=df_numeric_part.fillna(df_numeric_part.mean())
    return df

def map_cat_to_idx(chunk,cat_feat_=cat_feat,multihot_feat_=multihot_feat,cat_featureMap_=cat_featureMap,cat_featureIdx_beginAt_=cat_featureIdx_beginAt):
    """
    根据featureMap来进行映射
    效率上存在问题，1k数据&26个cat特征，cat_featureMap有88w，耗时约15s
    """
    chunk_=chunk.copy()
    for feat in cat_feat_:
        if feat in multihot_feat_:
            chunk_[feat]=chunk_[feat].apply(lambda x: [cat_featureMap_[feat].index(str(i))+cat_featureIdx_beginAt_[feat] for i in x])
        else:
            chunk_[feat]=chunk_[feat].apply(lambda x: cat_featureMap_[feat].index(str(x))+cat_featureIdx_beginAt_[feat])
    return chunk_

# df=normalize(df_head10,num_feat)
df=normalize(chunk,num_feat)
df=fill_numeric_NA(df,num_feat)
df=map_cat_to_idx(df)
df_featured=df
df_featured.head(5)


# ### 数据输入准备
# 分离label和两类特征

# In[9]:


df_idx_ = df_featured.copy()
df_idx_.head(5)
"{:.2f}KB".format(df_idx_.memory_usage().sum()/1e3)
label=df_idx_.pop("label").values
num_features=df_idx_[num_feat].values
cat_features=df_idx_[cat_feat].values
multihot_features=df_idx_[multihot_feat].values
onehot_features = df_idx_[list(set(cat_feat) - set(multihot_feat))].values

"label.shape",label.shape
"num_features.shape",num_features.shape
"onehot_features.shape",onehot_features.shape
"multihot_features.shape",multihot_features.shape

num_features[:5,:5]
onehot_features[:5,:5]
multihot_features[:5,:5]


# ### 直接来一手持久化

# In[34]:


import pickle

with open(os.path.join(base_dir,"num_features.pkl"),"wb") as fwb:
    pickle.dump(num_features,fwb)
    
with open(os.path.join(base_dir,"onehot_features.pkl"),"wb") as fwb:
    pickle.dump(onehot_features,fwb)
    
with open(os.path.join(base_dir,"multihot_features.pkl"),"wb") as fwb:
    pickle.dump(multihot_features,fwb)


# In[35]:


import pickle
with open(os.path.join(base_dir,"num_features.pkl"),"rb") as frb:
    num_features = pickle.load(frb)
    
with open(os.path.join(base_dir,"onehot_features.pkl"),"rb") as frb:
    onehot_features = pickle.load(frb)
    
with open(os.path.join(base_dir,"multihot_features.pkl"),"rb") as frb:
    multihot_features = pickle.load(frb)
num_features
onehot_features
multihot_features


# # Model

# 单独说明下tf里几个乘法
# - tf.tensordot
#     - 用`axes`参数用来表示两个待计算的元素哪一维是一致的
#         - (M,N) 和 (N,P) 相乘那就是 axes=1 表示第一个元素的第1维
#         ```python
#         >>> x1=np.arange(3).reshape(1,3)
#         x1: array([[0, 1, 2]])
#         >>> x2=np.arange(9).reshape(3,3)
#         x2: array([[0, 1, 2],
#                    [3, 4, 5],
#                    [6, 7, 8]])
#         >>> dotted=tf.tensordot(x1,x2,axes=1).numpy()
#         dotted: array([[15, 18, 21]])
#         >>> x1.shape,x2.shape,dotted.shape
#         res: ((1, 3), (3, 3), (1, 3))
#         ```
#         - (B,M,N) 和 (B,N,P) 相乘那就是 axes=[[2],[1]] 表示第一个元素的第2维和第二个元素的第1维
#         ```python
#         >>> x1=np.arange(3).reshape(1,1,3)
#         x1: array([[[0, 1, 2]]])
#         >>> x2=np.arange(9).reshape(1,3,3)
#         x2: array([[[0, 1, 2],
#                     [3, 4, 5],
#                     [6, 7, 8]]])
#         >>> dotted=tf.tensordot(x1,x2,axes=[[2],[1]]).numpy()
#         dotted: array([[[[15, 18, 21]]]])
#         >>> x1.shape,x2.shape,dotted.shape
#         res: ((1, 1, 3), (1, 3, 3), (1, 1, 1, 3))
#         ```
#     - 要求相乘的元素数据类型一致
# 
# - tf.keras.backend.dot
#     - 这个是做矩阵乘法的，需要满足两个输入是 MxQ QxN 的shape关系
#     - 要求相乘的元素数据类型一致
# ```python
# >>> tf.keras.backend.dot(tf.constant([[1,0,0]]), tf.constant([[1],[2],[3]])).numpy()
# array([[1]], dtype=int32)
# ```
# - tf.keras.layers.Dot
#     - 比较复杂，是通过初始化时的`axes`来控制计算时取哪个（或哪几个）维度去做点积——对应idx的元素乘起来最后加和
#     - 其实和`tf.tensordot`的`axes`是一样的
#     - tf官方文档的例子已经很清楚了 https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot

# 单独说明一下两个方法涉及 稀疏向量和稠密向量 计算的API（实际都是矩阵相乘）
# - tf.nn.embedding_lookup_sparse
#     - 相比于`tf.nn.embedding_lookup`就是多了个`combiner`来做合并
#         - 注：`tf.nn.embedding_lookup`和`tf.gather`逻辑是一样的，实现上前者的目的是为了支持更大向量（做了partition），见[文档](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
#     - 实际是查表然后把向量用`combiner`指定的方式“加和”
#     - `sp_ids`的values指定取哪个位置的emb向量,其indices不影响结果
#     - `sp_weights` 指定该emb向量乘某个权重，一般为None，即乘1
#     - `combiner` 选择加和的方式 "mean" "sqrtn" "sum"
#     - 注意它的`sp_ids`参数要求是SparseTensor，而SparseTensor构造的时候实际是二维的矩阵，如下示例演示"查前三个emb的均值"
#     ```python
#     >>> sp_ids_a=tf.sparse.SparseTensor(indices=[[0,0],[0,1],[0,2]],values=[0,1,2],dense_shape=[1,5])
#     >>> tf.sparse.to_dense(sp_ids_a).numpy()
#     array([[0, 1, 2, 0, 0]], dtype=int32)
#     >>> sp_ids_b=tf.sparse.SparseTensor(indices=[[0,0],[0,3],[0,4]],values=[0,1,2],dense_shape=[1,5])
#     >>> tf.sparse.to_dense(sp_ids_b).numpy()
#     array([[0, 0, 0, 1, 2]], dtype=int32)
#     >>> emb=tf.Variable(name='emb',initial_value=tf.keras.initializers.GlorotUniform(seed=2020)((cat_featureSize,4)))
#     >>> emb.numpy()
#     array([[-1.0650440e-03, -1.0466943e-03, -6.1871600e-04, -1.7025521e-03],
#            [-2.6915036e-04, -7.1870629e-05,  1.8417728e-03,  2.2394117e-03],
#            [-2.0872806e-03, -3.6849058e-04,  1.8430836e-03,  1.1817920e-03],
#            ...,
#            [ 1.2313656e-03,  1.6786391e-04, -2.4000034e-03,  1.8373653e-03],
#            [ 3.2466953e-04, -1.6028697e-03,  7.8225043e-05, -1.4092607e-03],
#            [ 2.4349121e-03, -7.7580218e-04, -1.6710395e-03,  5.3080427e-04]],
#           dtype=float32)
#     >>> tf.nn.embedding_lookup_sparse(params=emb,sp_ids=sp_ids_a,sp_weights=None,combiner="mean").numpy()
#     array([[-0.00114049, -0.00049569,  0.00102205,  0.00057288]],dtype=float32)
#     >>> tf.nn.embedding_lookup_sparse(params=emb,sp_ids=sp_ids_b,sp_weights=None,combiner="mean").numpy()
#     array([[-0.00114049, -0.00049569,  0.00102205,  0.00057288]],dtype=float32)
#     >>> (emb[0]+emb[1]+emb[2]).numpy()/3
#     array([-0.00114049, -0.00049569,  0.00102205,  0.00057288], dtype=float32)
#     ```
#     
# - `tf.sparse.sparse_dense_matmul`
#     - 这个是完全的矩阵相乘的逻辑，要求数据类型一致，要求是 MxN NxP 的shape关系
#     - 下面这个例子，相当于拿[0.5,1,2,0]和80w个emb向量做了点积
#     ```pythonn
#     >>> sp_ids_c=tf.sparse.SparseTensor(indices=[[0,0],[0,1],[0,2]],values=[0.5,1.,2.],dense_shape=[1,4])
#     >>> tf.sparse.to_dense(sp_ids_c).numpy()
#     array([[0.5, 1. , 2. , 0. ]], dtype=float32)
#     >>> tf.sparse.sparse_dense_matmul(sp_a=sp_ids_c,b=tf.transpose(emb)).numpy()
#     array([[-0.00281665,  0.0034771 ,  0.00227404, ..., -0.00401646,
#         -0.00128408, -0.00290043]], dtype=float32)
#     >>> tf.sparse.sparse_dense_matmul(sp_a=sp_ids_c,b=tf.transpose(emb)).numpy().shape
#     (1,885684)
#     ```

# ## 几种实现方案
# - [CTR预估模型：DeepFM/Deep&Cross/xDeepFM/AutoInt代码实战与讲解](https://zhuanlan.zhihu.com/p/109933924)
#     - FM一阶特征结果1维
#     - FM二阶特征结果1维 | 交叉乘积(emb1\*ebm2+emb2\*emb1)的结果(一个tensor)进行了reduce_sum
#     - NN特征结果1维 | 3个Dense(256)+1个Dense(1)
#     - 三个结果求和再过sigmoid
#     
# - [[带你撸论文]之Deep FM算法代码级精讲](https://zhuanlan.zhihu.com/p/109901389)
#     - FM一阶特征结果1维
#     - FM二阶特征结果emb维 | 保留交叉乘积的那个tensor
#     - NN特征结果N维 | 最后一个FC层的大小
#     - 三个结果concat再FC+sigmoid
#     
# - 连续特征处理
#     - 可以是给field一个emb，然后用numeric的值乘这个emb
#     - 也可以不过emb，直接concat在最后一FC前面（NN里的最后一个或者拼接后的最后一个FC）
# - multi-hot特征处理
#     - 一般就是取这几个emb的均值了
#     
# 论文里的图 & 原文解释
# - normal connection 是trainable的权重；
# - weight-1 connection 是权重为1的直连；
# - 中间用的activation是 relu 和 tanh
# ![image.png](attachment:image.png)
# 

# ## FMLayer

# In[18]:


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, emb_size=30, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.emb_size = emb_size
        self.kernel_emb = self.add_weight(name='kernel_emb', 
                                          shape=(self.feature_size, self.emb_size),
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_w = self.add_weight(name='kernel_w',
                                        shape=(self.feature_size,1),
                                        initializer='glorot_uniform',
                                        trainable=True)

    def build(self, input_shape):
        """
        重写build方法，在这里glorot初始化emb表
        改: FM的emb表和input_shape无关，还是不放在build里初始化了
        """
        super().build(input_shape)  
    
    def call(self, x):
        """
        x: 特征的索引矩阵;
        二阶特征: 索引查到各自的emb后，计算 <v1,v2>x1x2，这里实现时是 0.5*((emb1+emb2)^2 - emb1^2 - emb2^2)=emb1*emb2 
                 <emb1,emb2> != emb1*emb2 前者是内积得到的是标量，后者是element-wise的逐点相乘得到一个相同size的向量
        一阶特征: 直接根据idx查到权重w后求和，就是个LR
        """
        # 一阶特征
        x_1st_order = tf.reduce_sum(tf.nn.embedding_lookup(self.kernel_w,x),axis=1)
        # 二阶特征 | a*b+a*c+b*c
        x_emb = tf.nn.embedding_lookup(self.kernel_emb,x)
        x_square_of_sum = tf.square(tf.reduce_sum(x_emb,axis=1))  # (a+b+c)^2
        x_sum_of_square = tf.reduce_sum(tf.square(x_emb),axis=1)  # a^2 + b^2 + c^2
        x_2nd_order = 0.5*(x_square_of_sum - x_sum_of_square) # a*b+b*c+a*c
        x_2nd_order = tf.reduce_sum(x_2nd_order,axis=1,keepdims=True)
        fm_res = x_1st_order + x_2nd_order
        return fm_res

    def compute_output_shape(self, input_shape):
        """
        如 (100,10) 100个样本每个10个category特征，找到10个emb后取均值，返货结果就是 (100,emb_size)
        """
        return (input_shape[0], self.emb_size)
    
    @staticmethod
    def feature_idx_to_sparse_indices(arr):
        """
        把
        [[16,99,378,899],
         [12,89,103,500]]
        映射成:
        [[0,16],[0,99],[0,378],[0,899],
         [1,12],[1,89],[1,103],[1,500]] 
        用于给tf.sparse.SparseTensor提供indices参数，获得系数矩阵
        """
        return np.array([[[idx,v] for v in row] for idx,row, in enumerate(arr)]).reshape(arr.shape[0]*arr.shape[1],-1)

fml = FMLayer(feature_size=cat_featureSize,emb_size=4)


# ### FMLayer 计算拆解

# FMLayer的call()计算流程

# In[31]:


x.shape


# In[33]:


x=onehot_features
x[:5,:5]
# 一阶特征
x_1st_order = tf.reduce_sum(tf.nn.embedding_lookup(fml.kernel_w,x),axis=1)
# 二阶特征 | a*b+a*c+b*c
x_emb = tf.nn.embedding_lookup(fml.kernel_emb,x)
x_square_of_sum = tf.square(tf.reduce_sum(x_emb,axis=1))  # (a+b+c)^2
x_sum_of_square = tf.reduce_sum(tf.square(x_emb),axis=1)  # a^2 + b^2 + c^2
x_2nd_order = 0.5*(x_square_of_sum - x_sum_of_square) # a*b+b*c+a*c
x_2nd_order = tf.reduce_sum(x_2nd_order,axis=1,keepdims=True)
x_1st_order.shape
x_2nd_order.shape
fm_res = x_1st_order + x_2nd_order
x_1st_order.numpy()[:2,:2]
x_2nd_order.numpy()[:2,:2]
fm_res.numpy()[:2,:2]

print(">>> 直接用FMLayers计算")
fml(x).numpy()[:2,:2]


# 演示FM里二阶的计算
# - FM的理论把两个向量的内积作为权重，这两个向量就是每个cat特征的embedding
# - 这个思路来自于矩阵分解的理论，对于正定矩阵W有W=V*VT
# - 类似的也有attention里的QKV机制，认为Q、K的向量内积（点积）代表了二者的相似性
# - 更类似的还有CF里 MxN的User-Item矩阵拆分成 MxK的User矩阵 和 KxN的Item矩阵，User矩阵xItem矩阵得到User-Item预测矩阵，这里没有交互的项也得到了计算值，接下来迭代User矩阵和Item矩阵，让User-Item预测矩阵中已经有交互行为的项接近真实值；
# - FM的一阶和emb无关，就是普通的LR

# In[34]:


# x = cat_features
# x = tf.nn.embedding_lookup(params=fml.kernel,ids=x)
# 假设两个样本，各自有三个onehot特征，查出来的4维emb如下，每个样本都是[a,b,c] 且 a,b,c 均是4维emb向量
x = np.array([[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0],[10,10,10,10]],
              [[0.1,0.1,0.1,0.1],[0.1,0.2,0.3,0.4],[2.,2.,2.,2.]]])
x = tf.constant(x)
# (a+b+c)^2
x_square_of_sum = tf.square(tf.reduce_sum(x,axis=1))
# a^2 + b^2 + c^2
x_sum_of_square = tf.reduce_sum(tf.square(x),axis=1)
# FM二阶特征交叉 | a*b+a*c+b*c
x_2nd_order = (x_square_of_sum - x_sum_of_square)*0.5

x.numpy()
x_2nd_order.numpy()


# ## MultiFCLayer

# In[35]:


class MultiFCLayers(tf.keras.layers.Layer):
    def __init__(self,layer_units,**kwargs):
        super().__init__(**kwargs)
        self.layer_units = layer_units
        for idx,unit in enumerate(layer_units):
            if idx == 0 :
                continue
            self.add_weight(name=f"layer_{idx}",shape=(layer_units[idx-1],unit),initializer="glorot_uniform",trainable=True)
    
    def build(self, input_shape):
        super().build(input_shape)
        self.layer0=self.add_weight(name="layer_0",shape=(input_shape[-1],self.layer_units[0]),initializer="glorot_uniform",trainable=True)
        self.weights_dict={weight.name.split(":")[0]:weight for weight in self.weights}
        
        
    def call(self,x):
        for idx,_ in enumerate(self.layer_units):
            x = tf.matmul(x,self.weights_dict[f'layer_{idx}'])
        x = tf.nn.sigmoid(x)
        return x
    


# In[36]:


emb_size=8
nn_input_shape=(1000,onehot_features.shape[1]*emb_size+num_features.shape[1])
mfcl=MultiFCLayers([2,8,10])
mfcl.build(input_shape=nn_input_shape)
mfcl(np.random.random_sample(nn_input_shape).astype(np.float32))


# ## DeepFM 草稿1

# In[97]:


class DeepFM(tf.keras.Model):
    """
    注: 这里想支持多个multihot特征
    一个multihot特征如何去查embeding，其实现问题主要在于multihot特征是不定长的，ndarray是object类型不能直接转tensor的
    每个样本的multihot都是变长的，这最适合的解法就是表示为SparseTensor然后用tf.nn.embedding_lookup_sparse了
    多值特征从 (batch_size,1) 的object-arr变成 (batch_size,max_len) 的sparseTensor，经过lookup变成 (batch_size,emb_size)
    """
    def __init__(self, layer_units, feature_size, emb_size=30,dense_size=13,onehot_size=26,multihot_size=2):
        multihot_size=[] if multihot_size is None else multihot_size
        assert layer_units[-1]==1,"nn最后一层要输出1维，方便和fm的结果加和"
        super().__init__()
        self.dense_size=dense_size
        
        self.onehot_size=onehot_size
        self.multihot_size=multihot_size

        self.feature_size = feature_size
        self.emb_size = emb_size
        self.layer_units = layer_units
        
        # FM kernels
        self.fm_w = self.add_weight(name="fm_w",shape=(self.feature_size,1),initializer="glorot_uniform",trainable=True)
        self.fm_emb = self.add_weight(name="fm_emb",shape=(self.feature_size,self.emb_size),initializer="glorot_uniform",trainable=True)
        # NN kernels
        self.nn_w_b = []
        self.nn_input_size = emb_size*(onehot_size+multihot_size)+dense_size
        for idx,units in enumerate(self.layer_units):
            if idx==0:
                w = self.add_weight(name=f"nn_layer_{idx}_w",shape=(self.nn_input_size,units),initializer="glorot_uniform",trainable=True)                
            else:
                w = self.add_weight(name=f"nn_layer_{idx}_w",shape=(self.layer_units[idx-1],units),initializer="glorot_uniform",trainable=True)                
            b = self.add_weight(name=f"nn_layer_{idx}_b",shape=((1,)))
            self.nn_w_b.append([w,b])
      
    def _set_inputs(self,inputs):
        """
        这里不能调用 multihot_idx_to_sparse_tensor 会提示这是Eager模式的东西
        如果给multihot_idx_to_sparse_tensor加了 @tf.function，也不行，这时候会提示这个函数的输入不支持变长list的ndarray数据
        只能手动写了，手动写也不行，还是提示 ValueError: You should not pass an EagerTensor to `Input`. For example, instead of creating an InputLayer, you should instantiate your model and directly call it on your input.
        """
        dense=tf.convert_to_tensor(inputs[0])
        onehot=tf.convert_to_tensor(inputs[1])
        multihot=inputs[2]
        multihot_features_list=[multihot[:,i] for i in range(multihot.shape[1])]
        
        multihot_sp_list=[]
        for multihot in multihot_features_list:
            sp_values=[]
            sp_indices=[]
            max_len = 0
            for idx,row in enumerate(multihot):
                sp_values.extend(row)
                sp_indices.extend([[idx,i] for i in range(len(row))])
                max_len = max_len if len(row) <= max_len else len(row)
            multihot_sp_list.append(tf.sparse.SparseTensor(indices=sp_indices,values=sp_values,dense_shape=[len(multihot),max_len]))
        
#         return dense,onehot,multihot_sp_list
        return [tf.keras.Input(tensor=dense),tf.keras.Input(tensor=onehot)]+[tf.keras.Input(tensor=i,sparse=True) for i in multihot_sp_list]
        
    def calc_emb(self,inputs):
        dense=inputs[0]
        onehot=inputs[1]
        multihot_sparse_list=inputs[2:2+self.multihot_size]
        ##########
        # emb查询
        ##########
        # onehot查emb | None*26*K
        onehot_emb=tf.nn.embedding_lookup(params=self.fm_emb,ids=onehot)
        # multihot查emb | None*2*K
        if self.multihot_size > 0:
            multihot_embs = []
            for multihot_sparse in multihot_sparse_list:
                multihot_emb = tf.nn.embedding_lookup_sparse(params=self.fm_emb,sp_ids=multihot_sparse,sp_weights=None,combiner="mean")
                multihot_embs.append(multihot_emb)
            multihot_emb = tf.stack(multihot_embs,axis=1)
            # cat_emb | onehot+multihot的emb | None*28*K
            cat_emb = tf.concat([onehot_emb,multihot_emb],axis=1)
        else:
            cat_emb = onehot_emb
        return onehot_emb,multihot_emb,cat_emb
         
    def call(self,inputs,training=False):
        dense=inputs[0]
        onehot=inputs[1]
        multihot_sparse_list=inputs[2:2+self.multihot_size]
        onehot_emb,multihot_emb,cat_emb = self.calc_emb(inputs)

        ########
        # FM计算
        ########
        # FM 1st order | sum(Wi*Xi) 因为Xi=1所以直接就是 sum(Wi)
        fm_1st = tf.reduce_sum(tf.nn.embedding_lookup(params=self.fm_w,ids=onehot),axis=1) # None*1
        # FM 2nd order | 0.5*((a+b+c)^2-a^2-b^2-c^2)
        a = tf.square(tf.reduce_sum(cat_emb,axis=1)) 
        b = tf.reduce_sum(tf.square(cat_emb),axis=1)
        fm_2nd = 0.5*(a-b)  # None*K
        # 可以在这里把K维的结果加和，这样在x都是1的情况下就等价于所有两两相乘的内积加和
        # 也可以保留K维，在后面concat到一起过全连接层
        fm_2nd = tf.reduce_sum(fm_2nd,axis=1,keepdims=True) # None*1
        
        ########
        # NN计算 
        ########
        nn_inp = tf.concat([tf.reshape(cat_emb,(-1,self.emb_size*(self.onehot_size+self.multihot_size))),num_features],axis=1)
        nn_res = nn_inp
        for w,b in self.nn_w_b:
            nn_res=tf.nn.relu(tf.matmul(nn_res,w)+b)
        
        ############
        # DeepFM计算 
        ############
        deepfm_res=tf.nn.sigmoid(fm_1st+fm_2nd+nn_res)
        
#         return fm_1st,fm_2nd,nn_res 
        return deepfm_res

    @DeprecationWarning
    def _call(self,inputs,training=False):
        """
        不支持直接处理变长的list型的ndarray，也许RaggedTensor是个可能性？但是外部如何直接获得RaggedTensor？
        """
        dense=inputs[0]
        onehot=inputs[1]
        multihot_list=inputs[2:2+self.multihot_size]
        ##########
        # emb查询
        ##########
        # onehot查emb | None*26*K
        onehot_emb=tf.nn.embedding_lookup(params=self.fm_emb,ids=onehot)
        # multihot查emb | None*2*K
        if self.multihot_size > 0:
            multihot_embs = []
            for multihot in multihot_list:
                multihot_sparse = self.multihot_idx_to_sparse_tensor(multihot)
                multihot_emb = tf.nn.embedding_lookup_sparse(params=self.fm_emb,sp_ids=multihot_sparse,sp_weights=None,combiner="mean")
                multihot_embs.append(multihot_emb)
            multihot_emb = tf.stack(multihot_embs,axis=1)
            # cat_emb | onehot+multihot的emb | None*28*K
            cat_emb = tf.concat([onehot_emb,multihot_emb],axis=1).shape
        else:
            cat_emb = onehot_emb

        ########
        # FM计算
        ########
        # FM 1st order | sum(Wi*Xi) 因为Xi=1所以直接就是 sum(Wi)
        fm_1st = tf.reduce_sum(tf.nn.embedding_lookup(params=self.fm_w,ids=onehot),axis=1) # None*1
        # FM 2nd order | 0.5*((a+b+c)^2-a^2-b^2-c^2)
        a = tf.square(tf.reduce_sum(cat_emb,axis=1)) 
        b = tf.reduce_sum(tf.square(cat_emb),axis=1)
        fm_2nd = 0.5*(a-b)  # None*K
        # 可以在这里把K维的结果加和，这样在x都是1的情况下就等价于所有两两相乘的内积加和
        # 也可以保留K维，在后面concat到一起过全连接层
        fm_2nd = tf.reduce_sum(fm_2nd,axis=1,keepdims=True) # None*1
        
        ########
        # NN计算 
        ########
        nn_inp = tf.concat([tf.reshape(cat_emb,(-1,self.emb_size*(self.onehot_size+self.multihot_size))),num_features],axis=1)
        nn_res = nn_inp
        for w,b in self.nn_w_b:
            nn_res=tf.nn.relu(tf.matmul(nn_res,w)+b)
        
        ############
        # DeepFM计算 
        ############
        deepfm_res=tf.nn.sigmoid(fm_1st+fm_2nd+nn_res)
        
#         return fm_1st,fm_2nd,nn_res 
        return deepfm_res
    
    @tf.function
    def multihot_idx_to_sparse_tensor(self, multihot):
        """
        multihot: 多值特征从 (batch_size,1) 的object-arr变成 (batch_size,max_len) 的sparseTensor
        [[239,577,833,2834],
         [231,627,913],
         [],
         [19,455,733,1000,1020]]
         变成
        indices=[[0,0],[0,1],[0,2],[0,3],
                 [1,0],[1,1],[1,2],
                 [3,0],[3,1],[3,2],[3,3],[3,4]]
        values=[239,577,833,2834,231,627,913,19,455,733,1000,1020]
         的SparseTensor
        
        检验:
        idx=0
        multihot_features_list[0].shape
        tf.sparse.to_dense(sp).numpy().shape

        len(multihot_features_list[0][idx])
        len(tf.sparse.to_dense(sp).numpy()[idx])
        multihot_features_list[0][idx]
        tf.sparse.to_dense(sp).numpy()[idx]

        """
        
        sp_values=[]
        sp_indices=[]
        max_len = 0
        for idx,row in enumerate(multihot):
            sp_values.extend(row)
            sp_indices.extend([[idx,i] for i in range(len(row))])
            max_len = max_len if len(row) <= max_len else len(row)
        return tf.sparse.SparseTensor(indices=sp_indices,values=sp_values,dense_shape=[len(multihot),max_len])


# num+onehot 类型

# In[43]:


params={
    "layer_units":[4,4,1],
    "feature_size":cat_featureSize,
    "emb_size":4,
    "dense_size":13,
    "onehot_size":26,
    "multihot_size":0
}
M=DeepFM(**params)
res = M([num_features,onehot_features])
res.shape


# num+onehot+multihot 类型

# In[98]:


params={
    "layer_units":[4,4,1],
    "feature_size":cat_featureSize,
    "emb_size":4,
    "dense_size":13,
    "onehot_size":26,
    "multihot_size":2
}
M=DeepFM(**params)

# multihot_features_list=[multihot_features[:,i] for i in range(multihot_features.shape[1])]
# sp_list=[M.multihot_idx_to_sparse_tensor(multihot) for multihot in multihot_features_list]

# onehot_emb,multihot_emb,cat_emb = M.calc_emb([num_features,onehot_features]+sp_list)
# res = M([num_features,onehot_features]+sp_list)
# res.shape
# res_pred = M.predict([num_features,onehot_features]+sp_list)
# res_pred.shape


# 查emb

# In[227]:


onehot_emb=tf.nn.embedding_lookup(params=M.fm_emb,ids=onehot_features)
# multihot查emb
multihot_list = [multihot_features[:,i] for i in range(multihot_features.shape[1])]
multihot_embs = []
for multihot in multihot_list:
    multihot_sparse = multihot_idx_to_sparse_tensor(multihot)
    multihot_emb = tf.nn.embedding_lookup_sparse(params=M.fm_emb,sp_ids=multihot_sparse,sp_weights=None,combiner="mean")
    multihot_embs.append(multihot_emb)
multihot_emb = tf.stack(multihot_embs,axis=1)
if M.multihot_size > 0:
    cat_emb = tf.concat([onehot_emb,multihot_emb],axis=1)
else:
    cat_emb = onehot_emb
cat_emb.shape


# 算FM

# In[228]:


fm_1st = tf.reduce_sum(tf.nn.embedding_lookup(params=M.fm_w,ids=onehot_features),axis=1) # None*1
# FM 2nd order | 0.5*((a+b+c)^2-a^2-b^2-c^2)
a = tf.square(tf.reduce_sum(cat_emb,axis=1)) 
b = tf.reduce_sum(tf.square(cat_emb),axis=1)
fm_2nd = 0.5*(a-b)  # None*K
# 可以在这里把K维的结果加和，这样在x都是1的情况下就等价于所有两两相乘的内积加和
# 也可以保留K维，在后面concat到一起过全连接层
fm_2nd = tf.reduce_sum(fm_2nd,axis=1,keepdims=True) # None*1

fm_1st.shape
fm_2nd.shape


# 算NN

# In[234]:


nn_inp = tf.concat([tf.reshape(cat_emb,(-1,M.emb_size*(M.onehot_size+M.multihot_size))),num_features],axis=1)
nn_inp.shape
M.nn_input_size

nn_res = nn_inp
for l in M.nn_layers:
    nn_res=tf.matmul(nn_res,l)


# ## DeepFM 草稿2

# In[636]:


class DeepFM(tf.keras.Model):
    def __init__(self, layer_units, feature_size, emb_size=30,dense_size=13,onehot_size=26,multihot_size=0):
        super().__init__()
#         self.dense_idx=dense_size
#         self.onehot_idx=dense_idx+onehot_size
#         self.multihot_idx=onehot_idx+multihot_size
        self.feature_size = feature_size
        self.emb_size = emb_size
        self.layer_units = layer_units
        self.nn = MultiFCLayers(layer_units)
        self.fm = FMLayer(feature_size,emb_size)
        assert layer_units[-1]==1,"nn最后一层要输出1维，方便和fm的结果加和"
        nn_input_shape=(None,onehot_size*self.emb_size+dense_size)
        self.nn.build(nn_input_shape)


    
    def call(self,inputs,training=None):
        """
        注意inputs接受的参数顺序
        """
        dense=inputs[0]
        onehot=inputs[1]
        multihot = inputs[2]
        onehot=tf.cast(onehot,tf.int32)
        multihot=tf.cast(multihot,tf.int32)
        assert dense.shape[0]==onehot.shape[0]==multihot.shape[0],"三类特征的batchsize不一致"
        
        fm_res=self.fm(onehot)
        emb=tf.nn.embedding_lookup(self.fm.kernel_emb,onehot)
        nn_emb_part=tf.reshape(emb,(-1,onehot.shape[-1]*self.emb_size))
        nn_res=self.nn(tf.concat([dense,nn_emb_part],axis=1))
        
        return tf.nn.sigmoid(fm_res+nn_res) #fm_res,nn_res
        


# In[638]:


cat_features.shape
num_features.shape
inp=np.concatenate([num_features,cat_features],axis=1).astype(np.float32)
inp.shape
print(">>> 在Model里按索引取的dennse onehot multihot的shape依次如下:")
dense=inp[:,:M.dense_idx]
onehot=inp[:,M.dense_idx:M.onehot_idx]
multihot=inp[:,M.onehot_idx:M.multihot_idx]
dense.shape
onehot.shape
multihot.shape,"注意到multihot即使是空数组，也可以有shape"

call_res=M([dense,onehot,multihot]).numpy()
print(">>> Model.call")
call_res.shape
call_res[:10]

pred_res=M.predict([dense,onehot,multihot])
print(">>> Model.predict")
pred_res.shape
pred_res[:10]


# # 完整pipline

# ## DeepFM 接受SparseTensor

# In[51]:


class DeepFM(tf.keras.Model):
    """
    注: 这里想支持多个multihot特征
    一个multihot特征如何去查embeding，其实现问题主要在于multihot特征是不定长的，ndarray是object类型不能直接转tensor的
    每个样本的multihot都是变长的，这最适合的解法就是表示为SparseTensor然后用tf.nn.embedding_lookup_sparse了
    多值特征从 (batch_size,1) 的object-arr变成 (batch_size,max_len) 的sparseTensor，经过lookup变成 (batch_size,emb_size)
    """
    def __init__(self, layer_units, feature_size, emb_size=30,dense_size=13,onehot_size=26,multihot_size=2):
        multihot_size=[] if multihot_size is None else multihot_size
        assert layer_units[-1]==1,"nn最后一层要输出1维，方便和fm的结果加和"
        super().__init__()
        self.dense_size=dense_size
        
        self.onehot_size=onehot_size
        self.multihot_size=multihot_size

        self.feature_size = feature_size
        self.emb_size = emb_size
        self.layer_units = layer_units
        
        # FM kernels
        self.fm_w = self.add_weight(name="fm_w",shape=(self.feature_size,1),initializer="glorot_uniform",trainable=True)
        self.fm_emb = self.add_weight(name="fm_emb",shape=(self.feature_size,self.emb_size),initializer="glorot_uniform",trainable=True)
        # NN kernels
        self.nn_w_b = []
        self.nn_input_size = emb_size*(onehot_size+multihot_size)+dense_size
        for idx,units in enumerate(self.layer_units):
            if idx==0:
                w = self.add_weight(name=f"nn_layer_{idx}_w",shape=(self.nn_input_size,units),initializer="glorot_uniform",trainable=True)                
            else:
                w = self.add_weight(name=f"nn_layer_{idx}_w",shape=(self.layer_units[idx-1],units),initializer="glorot_uniform",trainable=True)                
            b = self.add_weight(name=f"nn_layer_{idx}_b",shape=((1,)))
            self.nn_w_b.append([w,b])
        
        self.inp_dense = tf.keras.Input((self.dense_size),dtype=tf.float64)
        self.inp_onehot = tf.keras.Input((self.onehot_size),dtype=tf.int32)
        self.inp_multihot_list = [tf.keras.Input((None,), sparse=True,dtype=tf.int32) for i in range(self.multihot_size)]
        self._set_inputs([self.inp_dense,self.inp_onehot]+self.inp_multihot_list)
      
    @tf.function
    def calc_emb(self,inputs):
        dense=inputs[0]
        onehot=inputs[1]
        multihot_sparse_list=inputs[2:2+self.multihot_size]
        ##########
        # emb查询
        ##########
        # onehot查emb | None*26*K
        onehot_emb=tf.nn.embedding_lookup(params=self.fm_emb,ids=onehot)
        # multihot查emb | None*2*K
        if self.multihot_size > 0:
            multihot_embs = []
            for multihot_sparse in multihot_sparse_list:
                multihot_sparse = tf.cast(multihot_sparse,tf.int32)
                multihot_emb = tf.nn.embedding_lookup_sparse(params=self.fm_emb,sp_ids=multihot_sparse,sp_weights=None,combiner="mean")
                multihot_embs.append(multihot_emb)
            multihot_emb = tf.stack(multihot_embs,axis=1)
            # cat_emb | onehot+multihot的emb | None*28*K
            cat_emb = tf.concat([onehot_emb,multihot_emb],axis=1)
        else:
            cat_emb = onehot_emb
        return onehot_emb,multihot_emb,cat_emb
         
    def call(self,inputs,training=False):
        dense=inputs[0]
        onehot=inputs[1]
        multihot_sparse_list=inputs[2:2+self.multihot_size]
        onehot_emb,multihot_emb,cat_emb = self.calc_emb(inputs)

        ########
        # FM计算
        ########
        # FM 1st order | sum(Wi*Xi) 因为Xi=1所以直接就是 sum(Wi)
        fm_1st = tf.reduce_sum(tf.nn.embedding_lookup(params=self.fm_w,ids=onehot),axis=1) # None*1
        # FM 2nd order | 0.5*((a+b+c)^2-a^2-b^2-c^2)
        a = tf.square(tf.reduce_sum(cat_emb,axis=1)) 
        b = tf.reduce_sum(tf.square(cat_emb),axis=1)
        fm_2nd = 0.5*(a-b)  # None*K
        # 可以在这里把K维的结果加和，这样在x都是1的情况下就等价于所有两两相乘的内积加和
        # 也可以保留K维，在后面concat到一起过全连接层
        fm_2nd = tf.reduce_sum(fm_2nd,axis=1,keepdims=True) # None*1
        
        ########
        # NN计算 
        ########
#         print(">>> cat_emb",cat_emb)
#         print(">>> dense",dense)
        nn_inp = tf.concat([tf.reshape(cat_emb,(-1,self.emb_size*(self.onehot_size+self.multihot_size))),dense],axis=1)
#         print(">>> nn_inp",nn_inp)
        nn_res = nn_inp
        for w,b in self.nn_w_b:
            nn_res=tf.nn.relu(tf.matmul(nn_res,w)+b)
        
        ############
        # DeepFM计算 
        ############
#         print(">>> nn_res",nn_res)
        deepfm_res=tf.nn.sigmoid(fm_1st+fm_2nd+nn_res)
#         print(">>> deepfm_res",deepfm_res)
#         return fm_1st,fm_2nd,nn_res 
        return deepfm_res

    @staticmethod
    def multihot_ragged_idx_to_sparse_tensor(multihot):
        """
        multihot: 多值特征从 (batch_size,1) 的object-arr变成 (batch_size,max_len) 的sparseTensor
        [[239,577,833,2834],
         [231,627,913],
         [],
         [19,455,733,1000,1020]]
         变成
        indices=[[0,0],[0,1],[0,2],[0,3],
                 [1,0],[1,1],[1,2],
                 [3,0],[3,1],[3,2],[3,3],[3,4]]
        values=[239,577,833,2834,231,627,913,19,455,733,1000,1020]
         的SparseTensor
        
        检验:
        idx=0
        multihot_features_list[0].shape
        tf.sparse.to_dense(sp).numpy().shape

        len(multihot_features_list[0][idx])
        len(tf.sparse.to_dense(sp).numpy()[idx])
        multihot_features_list[0][idx]
        tf.sparse.to_dense(sp).numpy()[idx]

        """
        
        sp_values=[]
        sp_indices=[]
        max_len = 0
        for idx,row in enumerate(multihot):
            sp_values.extend(row)
            sp_indices.extend([[idx,i] for i in range(len(row))])
            max_len = max_len if len(row) <= max_len else len(row)
#         dense_shape=[len(multihot),max_len]
#         return sp_indices,sp_values,dense_shape
        return tf.sparse.SparseTensor(indices=sp_indices,values=sp_values,dense_shape=[len(multihot),max_len])


# ### init

# In[52]:


params={
    "layer_units":[4,4,1],
    "feature_size":cat_featureSize,
    "emb_size":4,
    "dense_size":13,
    "onehot_size":26,
    "multihot_size":2
}
M=DeepFM(**params)


# ### Run Test

# In[53]:


multihot_features_list=[multihot_features[:,i] for i in range(multihot_features.shape[1])]
sp_tensor_list=[M.multihot_ragged_idx_to_sparse_tensor(multihot_ragged) for multihot_ragged in multihot_features_list]
merged_inp = [num_features,onehot_features]+sp_tensor_list


res_call = M(merged_inp).numpy()
"call res",res_call.shape
res_call[:10]
res_pred = M.predict(merged_inp)
"pred res",res_pred.shape
res_pred[:10]


# ### Preprocess
# 如何参考 [这个官方例子](https://www.tensorflow.org/tutorials/structured_data/feature_columns) 把特征预处理这块用 feature_column 的api搞定？
# 
# `tf.feature_column`这条路目前还是卡在了不能处理ragged-arr上 试了`tf.keras.experimental.SequenceFeatures`也不行

# In[ ]:


# 自行遍历构造featureMap
chunksize=1000
df_total_iter=pd.read_csv(data_fp, chunksize=chunksize)
num_feat=[]
cat_feat=[]
cat_featureMap={}
multihot_feat=["C27","C28"]
for idx,chunk in tqdm(enumerate(df_total_iter),total=60*10000/chunksize):
    ###############################
    # 随机构造两个multihot特征
    ###############################
    m1 = []
    for i in range(chunk.shape[0]):
        # 随机 0~100 长度的list，list内的元素是 0~2000 注意str化
        random_size = np.random.randint(low=0,high=100+1)
        m1.append(np.random.randint(low=0,high=2000+1,size=random_size).astype(str))
    chunk['C27'] = m1
    m2 = []
    for i in range(chunk.shape[0]):
        # 随机 0~20 长度的list，list内的元素是 0~100 注意str化
        random_size = np.random.randint(low=0,high=20+1)
        m2.append(np.random.randint(low=0,high=100+1,size=random_size).astype(str))
    chunk['C28'] = m2

    ###############################
    # 提取numeric特征和category特征
    ###############################
    _num_feat=[i for i in chunk.columns if i.startswith("I")]
    _cat_feat=[i for i in chunk.columns if i.startswith("C")]
    if len(num_feat) > 0:
        assert num_feat == _num_feat,f"I系特征不符，前{idx*chunksize}条为: {num_feat}，此后为{_num_feat}"
    else:
        num_feat = _num_feat
    if len(cat_feat) > 0:
        assert cat_feat == _cat_feat,f"C系特征不符，前{idx*chunksize}条为: {cat_feat}，此后为{_cat_feat}"
    else:
        cat_feat = _cat_feat
    
    #############################
    # category特征构造出featureMap
    #############################
    for feat in cat_feat:
#         features=list(chunk[feat].unique())
        features=list(np.unique(np.hstack(chunk[feat].values.flat)))
        features_ori=cat_featureMap.get(feat,[])
        cat_featureMap.update({feat: list(set(features+features_ori))})
    
print("三类特征列名:")
print("num_feat",",".join(num_feat))
print("cat_feat",",".join(cat_feat))
print("multihot_feat",",".join(multihot_feat))


for k,v in cat_featureMap.items():
    print(f"k:{k}, cnt:{len(v)}")

cat_featureSize=sum([len(v) for k,v in cat_featureMap.items()])
print("所有的category特征总计: {}".format(cat_featureSize))

before=0
cat_featureIdx_beginAt={}
for k,v in cat_featureMap.items():
    cat_featureIdx_beginAt.update({k:before})
    before += len(v)
print("各category特征的起始索引:")
cat_featureIdx_beginAt


# In[126]:


# 定义一部分特征处理
def normalize(df_inp,num_fields):
    """
    连续特征归一化 min-max
    """
    df=df_inp.copy()
    max_records=df[num_fields].apply(np.max,axis=0)
    min_records=df[num_fields].apply(np.min,axis=0)
    denominator=(max_records-min_records).apply(lambda x: x if x!=0 else 1e-4) 
    for f in num_fields:
        df[f] = df[f].apply(lambda x: np.abs(x-min_records[f])/denominator[f])
    return df

def fill_numeric_NA(df_inp,num_fields):
    """
    连续特征的NA填充 | 暂时直接用均值填充
    """
    df=df_inp.copy()
    df_numeric_part=df[num_fields]
    df[num_fields]=df_numeric_part.fillna(df_numeric_part.mean())
    return df

def map_cat_to_idx(chunk,cat_feat_=cat_feat,multihot_feat_=multihot_feat,cat_featureMap_=cat_featureMap,cat_featureIdx_beginAt_=cat_featureIdx_beginAt):
    """
    根据featureMap来进行映射
    效率上存在问题，1k数据&26个cat特征，cat_featureMap有88w，耗时约15s
    """
    chunk_=chunk.copy()
    for feat in cat_feat_:
        if feat in multihot_feat_:
            chunk_[feat]=chunk_[feat].apply(lambda x: [cat_featureMap_[feat].index(str(i))+cat_featureIdx_beginAt_[feat] for i in x])
        else:
            chunk_[feat]=chunk_[feat].apply(lambda x: cat_featureMap_[feat].index(str(x))+cat_featureIdx_beginAt_[feat])
    return chunk_


# In[ ]:


def train():
    with tf.GradientTape as g:
        pass


# ### Loop

# In[ ]:



for idx,chunk in tqdm(enumerate(df_total_iter),total=60*10000/chunksize):
    df=normalize(chunk,num_feat)
    df=fill_numeric_NA(df,num_feat)
    df=map_cat_to_idx(df)

    label=df.pop("label").values
    num_features=df[num_feat].values
    cat_features=df[cat_feat].values
    multihot_features=df_idx_[multihot_feat].values
    onehot_features = df_idx_[list(set(cat_feat) - set(multihot_feat))].values
    
    
        


# ## DeepFM 接受RaggedArr

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 手写梯度下降

# In[102]:


# 梯度下降方式求解平方根
import math
x=16
a=x//2
lr=0.01
cnt=0

while True:
    # 手动求导后得到梯度的解析式
    grad=2*(a*a-x)*2*a
    print(f"[根]:{a:.4f<}, [梯度]:{grad:.4f}, [更新后的根]:{a+grad*lr:.4f}")
    a -= grad*lr
    if abs(a*a - x) < 0.00000001:
        print("stop at:", a)
        break
    cnt += 1
    if cnt > 10:
        break


# In[ ]:





# In[ ]:





# # Train

# In[ ]:





# In[ ]:





# # feature_column

# feature_column处理单独一列

# In[ ]:


sequence_categorical_column_with_vocabulary_list


# In[125]:


na_fill_dict = {f:0 for f in num_feat}
na_fill_dict.update({f:"nan" for f in cat_feat})
chunk_fillna = chunk.fillna(na_fill_dict)
f="C28"
print(">>> featureMap:")
",".join(cat_featureMap[f])
# tfc=tf.feature_column.categorical_column_with_vocabulary_list(key=f,vocabulary_list=cat_featureMap[f])
tfc=tf.feature_column.sequence_categorical_column_with_vocabulary_list(key=f,vocabulary_list=cat_featureMap[f])
tfc=tf.feature_column.indicator_column(tfc)
print(">>> original:")
chunk_fillna[f]
print(">>> feature_columned:")
tf.keras.experimental.SequenceFeatures(tfc)(dict(chunk_fillna))


# In[123]:


tf.keras.layers.DenseFeatures
tf.keras.experimental.SequenceFeatures


# In[ ]:




