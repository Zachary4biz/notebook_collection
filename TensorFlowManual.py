#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[3]:


import tensorflow as tf


# # Tensorflow API解释

# In[4]:


import tensorflow as tf
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')


# ## tf.add & tf.nn.bias_add
# - [StackOverflow](https://stackoverflow.com/questions/43131606/whats-the-difference-of-add-methods-in-tensorflow)
# >tf.add is a general addition operation, while tf.nn.bias_add is to be used specifically for adding bias to the weights, which **raises an exception if the dtypes aren't same.**
# >
# >Unlike **tf.add**, the **type of bias is allowed to differ from value** in the case where both types are quantized.

# ## tf.expand_dims & tf.reshape

# In[62]:


a_ = [[[1,1],[2,2]],
      [[3,3],[4,4]]]
a_ = np.array(a_)
"a_",a_.shape

a = tf.Variable(a_,dtype=tf.float32)
"a",a.shape

a_tfreshape = tf.reshape(a,[2,2,2,-1])
"a_tfreshape",a_tfreshape.shape
a_tfexpand = tf.expand_dims(a,-1)
"a_tfexpand",a_tfexpand.shape

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("原始shape为")
    sess.run(tf.shape(a))
    print("-1 就是在最后一维上扩展一个")
    sess.run(tf.shape(tf.expand_dims(a,-1)))
    print("0 就是在第0维上扩展一个")
    sess.run(tf.shape(tf.expand_dims(a,0)))
    print("1 就是在第1维上扩展一个")
    sess.run(tf.shape(tf.expand_dims(a,1)))


# ## tf.stack
# - stack要求两个tensor维度必须是相同的，本来是 $[D_1,D_2,..D_n]$ 的共n个维度的俩tensor，stack后变成n+1个维度，多+1的那个维度为`2`，具体这个+1的维度`2`放在哪就由`axis=`决定，`axis=0`那这个`2`就放在索引0上
# 
# shape为(3,4,5)的两个tensor在不同axis上做stack
# - axis=0: (**2**,3,4,5)
# - axis=1: (3,**2**,4,5)
# - axis=2: (3,4,**2**,5)
# - axis=3: (3,4,5,**2**)
# 
# stack三个维度相同的tensor那就是把`3`添加在`axis`指定的索引位置上

# In[49]:


a = np.random.random([3,4,5])
b = np.random.random([3,4,5])
e = np.random.random([3,4,5])
c = tf.stack([a,b,e], axis=3)
a.shape
b.shape
c.shape


# ## tf.unstack
# 按`axis`指定的维度拆开，该维度取值是多少就拆成多少个tensor
# 
# shape为(3,4,5,2)的一个tensor在不同axis上做unstack
# - axis=0: [(4,5,2)]*3
# - axis=1: [(3,5,2)]*4
# - axis=2: [(3,4,2)]*5
# - axis=3: [(3,4,5)]*2

# In[57]:


m = np.random.random([3,4,5,2])
m.shape
tf.unstack(m)
tf.unstack(m,axis=0)
tf.unstack(m,axis=1)
tf.unstack(m,axis=2)
tf.unstack(m,axis=3)

# ?tf.unstack


# ## tf.reduce_mean
# tf.reduce_mean, reduce_sum, reduce_max就是计算某一个维度上的均值、加和、最值
# >tf.reduce_mean(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)
# - axis：
#     - axis=None, 求全部元素的平均值；
#     - axis=0, 列平均值；
#     - axis=1，行平均值。 
# - keep_dims：若值为True，可多行输出平均值。 
# - name：自定义操作名。 
# - ~~reduction_indices：axis的旧名，已停用。~~

# In[1]:


np.random.seed(2019)
a = np.random.randint(0,10,size=[2,3])
"a",a
a_var = tf.Variable(a)
a_var.shape
a.shape
tf.reduce_mean(a,axis=None).shape
tf.reduce_mean(a,axis=0).shape
with tf.Session() as sess:
    sess.run(tf.reduce_sum(a,axis=None))
    sess.run(tf.reduce_mean(a,axis=None))
    sess.run(tf.reduce_mean(a,axis=None)).shape
    sess.run(tf.reduce_mean(a,axis=0))
    sess.run(tf.reduce_mean(a,axis=0)).shape


# ## tf.transpose
# > tf.transpose(a, perm=None, name='transpose', conjugate=False)
# - a 需要转置的tensor
# - perm （permute）转置的形式
#     - `None` 表示把shape倒转过来，如[3,4]变成[4,3]，[1,2,3,4]变成[4,3,2,1]
#     - `list[int]类型` 里面的int表示原始维度的索引按list里的顺序来排列
#         - 如`[0,3,2,1]`表示原始的维度`3`放到第二个,`1`放到第四个（二、四维互换了）
#         - 如`[1,3,2,0]`表示转置后的，按数字作为索引把原始的维度按当前list里的顺序重新排列

# In[23]:


a = np.random.random([3,4,5,6])
a.shape
tf.transpose(a)
tf.transpose(a,[2,1,0,3])


# ## tf.truncated_normal
# 按指定均值、标准差生成正态分布的数据，并且做两倍标准差截断
# > tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) :
# - shape表示生成张量的维度
# - mean是均值 | 默认为0
# - stddev是标准差。 | 默认为1.0
# - seed随机数种子
# >
# >这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
# 
# tf里的随机数种子，可以设置到图级别也可以设置为op级别
# - 图级别：
# ```
# # at some graph
# tf.set_random_seed(2019)
# tf.truncated_normal([3,4],stddev=0.1)
# ```
# - op级别：
# ```
# tf.truncated_normal([3,4],stddev=0.1,seed=2019)
# ```
# 
# 类似的随机函数还有 `tf.random_uniform([3,4], -1, 1)` 生成-1到1的均匀分布的随机数
# >tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

# In[36]:


# truncated_normal 按正态分布生成数据，并且做标准差截断
with tf.Session() as sess:
    random_op = tf.truncated_normal([3,4],stddev=0.1,seed=2019)
    # random_op在一段程序里跑了三次，seed只控制程序每次相同位置生成时结果是一样的，而这三次则都不一样
    sess.run(random_op)
    sess.run(tf.cast(random_op,tf.int32))
    sess.run(tf.to_float(random_op,tf.int32))


# ## tf.while_loop

# In[17]:


i  = 0
n =10 

def cond(i, n):
    return i < n

def body(i, n):
    i = i + 1
    return i, n
i, n = tf.while_loop(cond, body, [i, n])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run([i,n])


# ## tf.slice & tf.gather
# - `tf.slice`: 按照指定的下标范围抽取连续区域的子集
#     - 用得少，一般直接用索引 `[a:b,c:d,:-5]` 这种方式直接取（更pythonic）
# - `tf.gather`: 按照指定的下标集合从axis=0中抽取子集，适合抽取不连续区域的子集
# 
# ### tf.slice
# > Note that tf.Tensor.getitem is typically a more pythonic way to perform slices, as it allows you to write foo[3:7, :-2] instead of tf.slice(foo, [3, 0], [4, foo.get_shape()[1]-2]).
# 
# 即`tf.slice(tensor,[3,0],[4,tensor.get_shape()[1]-2])`等价于`tensor[3:7,-2]`
# 
# ### tf.gather
# ```python
# tf.gather(
#     params,
#     indices,
#     validate_indices=None,
#     name=None,
#     axis=None,
#     batch_dims=0
# )
# ```

# In[29]:


input = np.array([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
input.shape
print("如下两个是一样的，因为第三维总共就三个元素，取0:3就是所有的都取了")
input[1:2,0:1]
input[1:2,0:1,0:3]
print(">>> sess res as follow:")
with tf.Session() as sess:
    sess.run(tf.slice(input, [1, 0, 0], [1, 1, 3])) # 等价于 input[1:2,0:1,0:3]
    # [[[3, 3, 3]]]

    sess.run(tf.gather(input, [0, 2]))
    # 
    # [[[1, 1, 1], [2, 2, 2]],
    #  [[5, 5, 5], [6, 6, 6]]]


# ## tf.cast
# 类型转换

# In[11]:


wordEmbedding = np.array([[0.8,0.9,0.6,0.5],[0.1,0.2,0.3,0.4]])
wordEmbedding
tensor = tf.cast(wordEmbedding,dtype=tf.float32,name='word2vec')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tensor)


# ## tf.nn.zero_fraction
# 计算为0的比例

# In[189]:


with tf.Session() as sess:
    tf.nn.zero_fraction([1,1,1,0]).eval()
    tf.nn.zero_fraction([1,1,0,0]).eval()


# ## tf.nn.embedding_lookup
# ```python
# tf.nn.embedding_lookup(
#     params,
#     ids,
#     partition_strategy='mod',
#     name=None,
#     validate_indices=True,
#     max_norm=None
# )
# ```

# In[16]:


emb = np.array([[1,2,3,4],[0.1,0.2,0.3,0.4],[10,20,30,40],[100,200,300,400]])
emb.shape
word_idx = [[0,1,2,1],[0,2,2,2]]

with tf.Session() as sess:
    sess.run(tf.nn.embedding_lookup(emb,word_idx))


# ## tf.nn.embedding_lookup_sparse
# - 可以参考下[这篇简书的文章](https://www.jianshu.com/p/f54eb9715609)
# ```python
# tf.nn.embedding_lookup_sparse(
#     params,
#     sp_ids,
#     sp_weights,
#     partition_strategy='mod',
#     name=None,
#     combiner=None,
#     max_norm=None
# )
# ```
# - `sp_weights`可以直接填`None`
# - 实际“查表”的时候就是用的`sp_ids`这个`sparseTensor`的`values`作为索引去`params`里查

# In[46]:


emb = np.array([[1,2,3,4],
                [0.1,0.2,0.3,0.4],
                [10,20,30,40],
                [100,200,300,400],
                [1000,2000,3000,4000]
               ])

word_idx = [[0,1,2,1],
            [0,2,2,2]]

word_idx_sp = tf.sparse.SparseTensor(indices=[[0, 0], [1, 0], [2, 0]],
                              values=[2,3,4],
                              dense_shape=[10, 1])
word_idx_w = tf.sparse.SparseTensor(indices=word_idx_sp.indices,
                                    values=tf.ones_like(word_idx_sp.values),
                                    dense_shape=word_idx_sp.dense_shape)
with tf.Session() as sess:
    sess.run(word_idx_sp)
    sess.run(tf.sparse.to_dense(word_idx_sp))
    sess.run(tf.nn.embedding_lookup_sparse(emb,sp_ids=word_idx_sp,sp_weights=None,combiner='mean'))
    sess.run(tf.nn.embedding_lookup_sparse(emb,sp_ids=word_idx_sp,sp_weights=word_idx_w,combiner='mean'))


# ## tf.nn.conv2d
# ```python
# tf.nn.conv2d(
#     input,
#     filter=None,
#     strides=None,
#     padding=None,
#     use_cudnn_on_gpu=True,
#     data_format='NHWC',
#     dilations=[1, 1, 1, 1],
#     name=None,
#     filters=None
# )
# ```
# 找到的解释：
# > tensorflow中的tf.nn.conv2d函数，实际上相当于用filter，以一定的步长stride在image上进行滑动，计算重叠部分的内积和，即为卷积结果
# 
# 官方文档：
# > input tensor of shape:  
# > - [`batch`, **in_height**, **in_width**, **in_channels**]
# >
# > filter / kernel tensor of shape:   
# >- [**filter_height**, **filter_width**, **in_channels**, `out_channels`]
# >
# >this op performs the following:
# >- Flattens the filter to a 2-D matrix with shape:
# >    - [**filter_height** * **filter_width** * **in_channels**, `output_channels`].
# >- Extracts image patches from the input tensor to form a virtual tensor of shape:
# >    - [`batch`, out_height, out_width, **filter_height** * **filter_width** * **in_channels**].
# >- For each patch, right-multiplies the filter matrix and the image patch vector.

# In[ ]:


tf.on


# In[102]:


emb = np.array([[1,2,3,4,5,6],
                [0.1,0.2,0.3,0.4,0.5,0.6],
                [10,20,30,40,50,60],
                [100,200,300,400,500,600]])
# word_idx = [[0,1,2,1],[0,2,2,2]]
word_idx = [[0,1,2]]
embeddedWords = tf.cast(tf.nn.embedding_lookup(emb,word_idx),dtype=tf.float32)
embeddedWordsExpanded = tf.expand_dims(embeddedWords, -1)

embeddedWordsExpanded.shape
filterSize = 2 # 卷积核大小
embeddingSize = 6 # 词向量维度
in_channels =1 # 输入的通道
numFilters = 4 # 卷积核的个数
sequenceLength = len(word_idx[0]) # 句子长度，一般要padding
filterShape = [filterSize, embeddingSize, in_channels, numFilters] # 构建conv2d使用的filter参数
# W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1,dtype=tf.float64), name="W",dtype=tf.float64)
# b = tf.Variable(tf.constant(0.1, shape=[numFilters],dtype=tf.float64), name="b",dtype=tf.float64)
# W = tf.convert_to_tensor(tf.truncated_normal(filterShape, stddev=0.1), name="W") # 正态分布随机初始化
W = tf.convert_to_tensor(tf.ones(filterShape), name="W") #
b = tf.convert_to_tensor(tf.constant(0.1,shape=[numFilters]),name="b")
conv = tf.nn.conv2d(input=embeddedWordsExpanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#     embeddedWords.eval()
    tf.shape(embeddedWordsExpanded).eval()
    tf.shape(W).eval()
    tf.shape(conv).eval()
    embeddedWordsExpanded.eval()
#     tf.shape(W).eval()
    print(">>> 每个卷积核都初始化为相同的权重W，目前按1填充")
    W.eval()
    print(">>> 偏置 b:")
    b.eval()
    print(">>> 1+2+3+4+5+6=21,0.1+..0.6=2.1,每个卷积的结果为23.1")
    conv.eval()
    


# ## tf.argmax & tf.nn.softmax
# 

# In[106]:


a = [1,2,3,4]
a = [float(i) for i in a]
with tf.Session() as sess:
    tf.argmax(a).eval()
    tf.nn.softmax(a).eval()


# In[173]:


sigmoid(1)
sigmoid(0)


# ## cross_entropy
# **「注」：** tf的这批API，都是在内部做了sigmoid
# - 测试用例是 `labels=[1,1,0,1],logits=[1,1,0,1]`
#     - 内部是对`loigts`做了个`sigmoid`，即真正计算的是 `[1,1,0,1]` 和 `[0.731,0.731,0.5,0.731]`之间的CE**
#     - `sigmoid(1)=0.731, sigmoid(0)=0.5`
#     
# 
# 一些参考
# - 参考这篇[博客Tensorflow损失函数详解](https://sthsf.github.io/wiki/Algorithm/DeepLearning/Tensorflow%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86---%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E8%AF%A6%E8%A7%A3.html)
# 
# - 这篇[简书文章](https://www.jianshu.com/p/cf235861311b)结构更清晰
# 
# - CE公式
#     - $H(X=x)=-\sum_x p(x)logq(x)$
# 
# - logloss
#     - $logloss = - \sum_{i=1}^n(\frac{\hat{y_i}}{n}log(y_{pred})+\frac{1-\hat{y_i}}{n}log(1-y_{pred}))$
# 
# - 单看一条样本的logloss
#     - $logloss = ylog(\hat{y}) + (1-y)log(1-\hat{y})$
# 

# In[175]:


class Data():
    # 每一行可有多个1,如一张图既有 label_桌子 又有 label_椅子
    multi_hot_labels=np.array([[1,0,0],
                               [0,1,0],
                               [0,0,1],
                               [1,1,0],
                               [0,1,0]],dtype=np.float32)
    
    # 每一行只有一个1,如一张图只能有 label_桌子 不能有 label_椅子
    one_hot_labels=np.array([[1,0,0],
                             [0,1,0],
                             [0,0,1],
                             [1,0,0],
                             [0,1,0]],dtype=np.float32)
    

    logits=np.array([[12,3,2],
                     [3,10,1],
                     [1,2,5],
                     [4,6.5,1.2],
                     [3,6,1]],dtype=np.float32)
    
    
Data.multi_hot_labels
Data.one_hot_labels
Data.logits


# ### 如下是tensorflow计算CE的公式化简
# 
# 「注」：这里已经把`sigmoid`考虑进去了，所以输入的时候`pred`就不要进行`sigmoid`了
# - For brevity, let x = logits, z = labels. The logistic loss is
# ```python
# z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
# = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
# = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
# = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
# = (1 - z) * x + log(1 + exp(-x))
# = x - x * z + log(1 + exp(-x))
# ```
# - For x < 0, to avoid overflow in exp(-x), we reformulate the above
# ```python
# x - x * z + log(1 + exp(-x))
# = log(exp(x)) - x * z + log(1 + exp(-x))
# = - x * z + log(1 + exp(x))              
# ```
# - Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
# ```python
# max(x, 0) - x * z + log(1 + exp(-abs(x)))
# ```

# In[156]:


from math import log,exp
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 完全直接的CE，输入的是label和外部做好sigmoid的prediction
def exact_ce(pred,label):
    return -label*np.log(y_pred)-(1-label)*np.log(1-y_pred)

# tf化简公式（内部做了sigmoid，已化简掉了）
def ce_as_tf(pred,label):
    return max(pred, 0) - pred * label + log(1 + exp(-abs(pred)))

# 按计算公式计算（内部做了sigmoid）
def manual_formula(pred,label):
    y_pred = sigmoid(pred)
    E1 = -label*np.log(y_pred)-(1-label)*np.log(1-y_pred)
    return E1

ce_as_tf(1,1)
manual_formula(0.7,0.7) == ce_as_tf(0.7,0.7)


# ### tf.nn.sigmoid_cross_entropy_with_logits
# - 这个函数的输入是logits和labels，logits就是神经网络模型中的 W * X矩阵，注意**不需要经过sigmoid**
# - 可用于各类别独立但不排斥的情况：如图片可以既有桌子又有凳子
# 

# In[169]:


# 5个样本三分类问题，且一个样本可以同时拥有多类
print(manual_formula(pred=Data.logits,label=Data.multi_hot_labels))     # 按计算公式计算的结果

with tf.Session() as sess:
    tf.nn.sigmoid_cross_entropy_with_logits(logits=Data.logits,labels=Data.multi_hot_labels).eval()
    tf.nn.sigmoid_cross_entropy_with_logits(logits=Data.logits,labels=Data.one_hot_labels).eval()


# ### tf.nn.weighted_cross_entropy_with_logits
# - 是`sigmoid_cross_entropy_with_logits`的拓展版，多支持一个`pos_weight`参数，在传统基于sigmoid的交叉熵算法上，**正样本算出的值乘以某个系数。**
# 
# ```python
# tf.nn.weighted_cross_entropy_with_logits(
#     labels=None,
#     logits=None,
#     pos_weight=None,
#     name=None,
#     targets=None
# )
# ```

# In[171]:


# pos_weight = np.ones_like(logits.shape[0])
pos_weight = np.zeros_like(Data.logits.shape[0]) # 权重统一为0
with tf.Session() as sess:
    tf.nn.weighted_cross_entropy_with_logits(Data.multi_hot_labels,Data.logits, pos_weight, name=None).eval()
    tf.nn.weighted_cross_entropy_with_logits(Data.one_hot_labels,Data.logits, pos_weight, name=None).eval()


# ### ~~tf.nn.softmax_cross_entropy_with_logits~~ (Deprecated)
# ### tf.nn.softmax_cross_entropy_with_logits_v2
# - 为了效率此函数内部执行softmax，输入logits时不要计算softmax
# - While the **classes are mutually exclusive**, their probabilities need not be. All that is required is that **each row of labels is a valid probability distribution**
# - Note that to avoid confusion, it is required to pass only named arguments to this function.
# 

# In[174]:


with tf.Session() as sess:
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Data.multi_hot_labels,logits=Data.logits).eval()
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Data.one_hot_labels,logits=Data.logits).eval()


# ### tf.nn.sparse_softmax_cross_entropy_with_logits
# 注意labels和logits的shape
# - labels是第几类的索引
# ```python
#    [2,
#     1,
#     1,
#     2]
# ```
# - logits是
# ```python
#     [[ 12, 4,   4, 22],
#      [6.5, 2, 3.3,  7],
#      [2.5, 9, 8.3,6.7],]
# ```
# >如果两个都是Rank1会报错： Rank of labels (received 1) should equal rank of logits minus 1 (received 1)
# 
# ```python
# tf.nn.sparse_softmax_cross_entropy_with_logits(
#     _sentinel=None,
#     labels=None,
#     logits=None,
#     name=None
# )
# ```

# In[185]:


# Data.multi_hot_labels
# tf.argmax(Data.multi_hot_labels,axis=-1).eval() # multi_hot（支持一图多类）的label做aargmax也没有意义
Data.one_hot_labels
Data.logits
label_rank1 = tf.argmax(Data.one_hot_labels,axis=-1)
logits_rank1 = tf.argmax(Data.logits,axis=-1)
with tf.Session() as sess:
    label_rank1.eval()
    logits_rank1.eval()
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_rank1,logits=Data.logits).eval()
#     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=,logits=Data.logits).eval()


# ### 对比

# In[187]:


with tf.Session() as sess:
    # softmax
    label_rank1=tf.argmax(Data.one_hot_labels,axis=-1)
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_rank1,logits=Data.logits).eval()
    
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Data.one_hot_labels,logits=Data.logits).eval()
    
    # sigmoid
    tf.nn.sigmoid_cross_entropy_with_logits(logits=Data.logits,labels=Data.one_hot_labels).eval()

    pos_weight=np.ones_like(Data.logits.shape[0]) # 权重统一为1
    tf.nn.weighted_cross_entropy_with_logits(Data.one_hot_labels,Data.logits, pos_weight, name=None).eval()


# In[ ]:




