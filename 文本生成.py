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
import numpy as np
from zac_pyutils.ExqUtils import zprint


# In[2]:


import tensorflow as tf
import numpy as np
import json
import re
import itertools
import pickle
import time


# In[3]:


# 允许GPU渐进占用
sess_conf = tf.ConfigProto()
sess_conf.gpu_options.allow_growth = True  # 允许GPU渐进占用
sess_conf.allow_soft_placement = True  # 把不适合GPU的放到CPU上跑
with tf.Session(config=sess_conf) as sess:
    print(sess)


# In[4]:


# sess.run(.. options=run_opt)可以在OOM的时候提供当前已经声明了的变量
run_opt = tf.RunOptions()
run_opt.report_tensor_allocations_upon_oom = True


# # 试试看能不能yield方式构造出单词索引
# - 要跟后面去正文使用相同的 `load_f` 加载方式（相同的预处理）
# 👌已完成

# In[5]:


# data_name = "labeled_timeliness_region_taste_emotion_sample.json.bak.head1k"
data_name = "labeled_timeliness_region_taste_emotion_sample.json.bak"
fp = "/home/zhoutong/NLP/data/{}".format(data_name)
result_set_fp = "/home/zhoutong/NLP/data/{}_char2idx".format(data_name)
coded_article_fp = "/home/zhoutong/NLP/data/{}_encoded_article.pkl".format(data_name)

"fp: ",fp
"result_set_fp: ", result_set_fp
"coded_article_fp: ", coded_article_fp


# In[6]:


def load_f(fp_inp):
    with open(fp_inp,"r") as f:
        for line in f:
            title = json.loads(line)['title']
            text = json.loads(line)['text']
            text = re.sub("[\\n]+", "\\n",text)
            yield text


# In[ ]:



def transform(text_inp):
 """
 这里是把各个标点符号都前后加上空格分开，不确定这样是否可以增加文本生成时对标点的准确表示
 理论上在建立索引的时候表征过的元素（例如"\n"索引为0）就有可能性
 但是不分开，直接把 "you!"(idx=11) 当作一个新的整体而不是 "you"(idx=9) 和 "!"(idx=10) 可能也行
 """
 for t in ["\\n",", "]:
     text_inp = re.sub(t, " "+t+" ",text_inp)
 text_inp = re.sub("\. "," . ",text_inp) # "." 不好直接放在循环中一起做，规矩不太一样单独做了 
 return text_inp


text_g = load_f(fp)
result_set = set()
while True:
 chunk = list(itertools.islice(text_g,10000))
 if len(chunk) > 0:
     for text in chunk:
         # 不使用transform
         # text = transform(text)
         result_set.update(text.replace("\n"," \n ").strip().split(" "))
 else:
     result_set = [i for i in result_set if i != ""]
     break


import pickle
result_set_d = dict([(word,idx) for idx,word in enumerate(result_set)])
with open(result_set_fp+".pickle","wb+") as f:
 pickle.dump(result_set_d,f)


# In[ ]:


with open(result_set_fp+".pickle","rb+") as f:
    word2idx_dict = pickle.load(f)


# In[ ]:


list(itertools.islice(word2idx_dict.items(),10))


# # 实验性质 | 看看出来的结果对不对

# In[ ]:


from collections import deque

text_g = load_f(fp)
wordsIdx = deque()
stopCnt = 0
while True:
    chunk = list(itertools.islice(text_g,10000))
    if len(chunk) > 0:
        for text in chunk:
            words = text.replace("\n"," \n ").strip().split(" ")
            words = [i for i in words if i != ""]
            wordsIdx.append([word2idx_dict[w] for w in words])
            print(">>>", words[:10])
            for i in list(itertools.islice(wordsIdx,10)):
                print(i[:10])  # 每次都打印wordsIdx的top10段落的top10个词
            stopCnt += 1
            assert stopCnt<=5
    else:
        result_set = [i for i in result_set if i != ""]
        break


# # 文章替换成word索引
# - 这里每篇文章都是一个单独的数组`append`到`wordsIdx`里
# - 这个二维数组存npy文件太大了，转成二维list存
#     - npy: 5.1G | deque_pkl: 3.2G | list_pkl: 3.2G
#     - 直接以deque存和转成list存占用空间相同
# 

# In[ ]:


from collections import deque

text_g = load_f(fp)
wordsIdx = deque()
with tqdm() as pbar:
    while True:
        chunk = list(itertools.islice(text_g,10000))
        if len(chunk) > 0:
            for text in chunk:
                words = text.replace("\n"," \n ").strip().split(" ")
                words = [i for i in words if i != ""]
                words_idx = ([word2idx_dict[w] for w in words]+[-1]*1024)[:300]  # 每篇文章最多取1024个词
                wordsIdx.append(words_idx)
                pbar.update(1)
        else:
            break
with open(coded_article_fp,"wb+") as fwb:
    pickle.dump(list(wordsIdx),fwb)


# In[ ]:


with open(coded_article_fp,"rb+") as frb:
    coded_article = pickle.load(frb)


# # CharRNN 基于字符

# ## 搭建模型

# ### 输入层

# In[5]:


def build_inputs(batch_size, num_steps):
    '''
    构建输入层
    
    batch_size: 每个batch中的序列个数
    num_steps: 每个序列包含的字符数
    '''
    inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='targets')
    
    # 加入keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob


# ### LSTM
# - `BasicLSTMCell` 替换为 `LSTMCell` 
# - LSTM需要知道 `batch_size` 只是用来做全零初始化时需要知道维度

# In[6]:


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' 
    构建lstm层
        
    keep_prob
    lstm_size: lstm隐层中结点数目
    num_layers: lstm的隐层数目
    batch_size: batch_size

    '''
    def construct_cell(node_size):
        # 构建一个基本lstm单元
        lstm = tf.nn.rnn_cell.LSTMCell(node_size)
        # 添加dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    # 堆叠
    cell = tf.nn.rnn_cell.MultiRNNCell([construct_cell(lstm_size) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


# ### 输出层
# - `tf.concat(1,lstm_output)` 替换为 `tf.concat(lstm_output,1)`

# In[7]:


def build_output(lstm_output, in_size, out_size):
    ''' 
    构造输出层
        
    lstm_output: lstm层的输出结果
    in_size: lstm输出层重塑后的size
    out_size: softmax层的size
    
    '''

    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
    # tf.concat的结果是[1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output, 1) # tf.concat(concat_dim, values)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])
    tf.summary.histogram('seq_output_reshape',x)
    
    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    tf.summary.histogram("softmax_w",softmax_w)
    tf.summary.histogram("softmax_b",softmax_b)
    
    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')
    tf.summary.histogram('pred',out)
    
    return out, logits


# ### 误差
# - `softmax_cross_entropy_with_logits` 替换为 `softmax_cross_entropy_with_logits_v2`

# In[8]:


def build_loss(logits, targets, lstm_size, num_classes):
    '''
    根据logits和targets计算损失
    
    logits: 全连接层的输出结果（不经过softmax）
    targets: targets
    lstm_size
    num_classes: vocab_size
        
    '''
    
    # One-hot编码
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss


# ### 优化器

# In[9]:


def build_optimizer(loss, learning_rate, grad_clip):
    ''' 
    构造Optimizer
   
    loss: 损失
    learning_rate: 学习率
    
    '''
    
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    for g in grads:
        tf.summary.histogram(g.name, g)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


# ### 模型
# - 使用 `placeholder` 替代固定的size和steps
# - 内部新建一张计算图而不是使用reset后的默认计算图
# - 增加summary

# In[45]:


class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, summary_path=None, sampling=False):
    
        batch_size, num_steps = batch_size, num_steps
        
        # 新建一张图
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            # 输入层
            self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

            # LSTM层
            cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

            # 对输入进行one-hot编码
            x_one_hot = tf.one_hot(self.inputs, num_classes)

            # 运行RNN
            outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
            self.final_state = state

            # 预测结果
            self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

            # Loss 和 optimizer (with gradient clipping)
            self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
            self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

            # summary
            tf.summary.scalar("loss", self.loss)
            #    lstm的variables在dynamic_run之后才会有值不然是空的list
            for idx,tensor in enumerate(cell.variables):
                if idx % 2 == 0:
                    _ = tf.summary.histogram(f"lstm_kernel_{idx}",tensor)
                else:
                    _ = tf.summary.histogram(f"lstm_bias_{idx}",tensor)
            self.merge_summary = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(summary_path, self.graph) if summary_path is not None else None
        


# ## 文本编码

# ### 数据指定

# In[11]:


def load_f(fp_inp):
    with open(fp_inp,"r") as f:
        for line in f:
            yield line

data_name = "anna.txt"
fp = "/home/zhoutong/NLP/data/{}".format(data_name)
result_set_fp = "/home/zhoutong/NLP/data/{}_char2idx".format(data_name)
coded_article_fp = "/home/zhoutong/NLP/data/{}_encoded_article.npy".format(data_name)

"fp: ",fp
"result_set_fp: ", result_set_fp
"coded_article_fp: ", coded_article_fp


# ### char2idx

# In[ ]:


text_g = load_f(fp)
result_set = set()
while True:
    chunk = list(itertools.islice(text_g,10000))
    if len(chunk) > 0:
        for text in chunk:
            # 不使用transform
            # text = transform(text)
            result_set.update(list(text))
    else:
        result_set = [i for i in result_set if i != ""]
        break


import pickle
result_set_d = dict([(word,idx) for idx,word in enumerate(result_set)])
with open(result_set_fp+".pickle","wb+") as f:
    pickle.dump(result_set_d,f)


# In[12]:


with open(result_set_fp+".pickle","rb+") as frb:
    char2idx = pickle.load(frb)
list(itertools.islice(char2idx.items(),10))
len(char2idx)


# ### idx2char

# In[13]:


idx2char={v:k for k,v in char2idx.items()}
list(itertools.islice(idx2char.items(),10))
len(idx2char)


# ### encoded (doc2idx)

# In[24]:


with open(fp,"r+") as fr:
    text = fr.read()

encoded = np.array([char2idx[c] for c in tqdm(text)])
np.save(coded_article_fp,encoded)  # 14G


# In[14]:


encoded = np.load(coded_article_fp).astype(np.float32)
encoded.shape
encoded[:10]


# ### get_batches函数

# In[15]:


def get_batches(encoded, batch_size, n_steps, verbose=False):
    chunk_len = batch_size*n_steps 
    n_chunk = int(len(encoded)/chunk_len)
    arr = encoded[:chunk_len*n_chunk]  # 截取整数倍的batch_size
    arr = arr.reshape((batch_size,-1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:,1:], y[:, 0]  # 这里应该有问题，最后y[:, 0]应该改成从取后一个才对为什么是又从0开始取
        yield x, y


# #### 以下是对get_batches函数的一个验证
# 这里其实是把整个文本语料按「字符」作为单位切分batch，完全舍弃了「词」的概念
# 
# 例如"I come from China"进行get_batches
# - `batch_size=3,n_steps=4` 说明这个batch里**有3个样本（句子），每个样本时间步长（字符数）是4**
# - 这时会计算这一共是多少个字符：3x4=12
# - 再计算整个句子支持多少个batch`n_chunk = int(len(encoded)/chunk_len)`，把余数去掉
# - 此后每次都用`[:, n:n+n_steps]`来迭代取一个batch的数据
# - 这个例句中刚好到'I come from '是12，后面的就被当余数去掉了
# - 得到的batch如下示例

# In[119]:


testStr = "I am from Chaoyang Beijing China"
batchSize=3
nSteps=4
print(f">>> 测试文本够「{len(testStr)//(batchSize*nSteps)}」个chunk，余下被截断丢弃了")
actual_used = testStr[:len(testStr)//(batchSize*nSteps)*(batchSize*nSteps)]  # 实际使用的文本部分
print(f"  -所以实际使用的测试文本是:'{actual_used}'")
print(f"  -丢弃的部分是          :'{testStr[len(testStr)//(batchSize*nSteps)*(batchSize*nSteps):]}'")
actual_used_reshaped = np.array(list(actual_used)).reshape((batchSize,-1))
print(f">>> get_batches函数里对截断后的arr还做了个`reshape((batchSize,-1))`，效果是:{actual_used_reshaped.shape}\n",actual_used_reshaped)
print("这样后面在对arr取索引 [:, n:n+n_steps] 时，其实是：第0个batch是从每行都取第0批的 n_steps 个元素")
print("这样看起来一个batch里的几个训练样本（训练sequence）之间并不是连续的，但是并不影响，样本内的sequence是连续的就行（即样本还是正确顺序的字符）")

print("\n*****这里y取的应有问题，每个训练样本的最后一个y好像是错的*****")
for idx,(x,y) in enumerate(get_batches(np.array(list(testStr)),batch_size=3,n_steps=4)):
    print(f"\n>>> 在第{idx}个batch里")
    print(f"x:\n",x)
    print(f"y (x的字符往后延一个):\n",y)


# In[ ]:


text_g = load_f(fp)
def get_batches(text_generator, batch_size, time_step, verbose=False):
    X_batch,Y_batch = [],[]
    X_verbose,Y_verbose = [],[]
    chunk = list(itertools.islice(text_generator, batch_size))
    for text in chunk:
        # 每次生成一篇文章的样本都从from_idx开始取time_step个字符
        from_idx=np.random.randint(len(text)-time_step-1) # from_idx用随机数,最后的-1是为了把最后一个字符留给Y
        text_X = text[from_idx:from_idx+time_step]
        text_Y = text[from_idx+1:from_idx+time_step+1]
        if verbose:
            X_verbose.append(text_X)
            Y_verbose.append(text_Y)
        X_batch.append([char2idx[char] for char in text_X])
        Y_batch.append([char2idx[char] for char in text_Y])
    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    if verbose:
        return X_batch, Y_batch, np.array(X_verbose), np.array(Y_verbose)
    else:
        return X_batch, Y_batch

get_batches(text_g,20,5,True)


# ## 超参

# In[37]:


batch_size = 100         # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.01    # Learning rate
keep_prob = 0.5         # Dropout keep probability


# ## 训练

# In[38]:


epochs = 40
# 每n轮进行一次变量保存
save_every_n = 200
summary_path = './tmp/tensorboard_anna'
base_model_path = "./tmp/lstm_anna/i{}_l{}.ckpt"

model = CharRNN(len(char2idx), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate, summary_path=summary_path)

# 索引转成字符
def _tochar(i):
        return idx2char[i]
_tochar_vec = np.vectorize(_tochar)
# 取输入的x y preds的字符映射结果的第一个样本
def get_sample_char(x,y,preds,verbose=False):
    # preds先reshape一下
    preds_reshape = preds.reshape(batch_size,num_steps,len(char2idx))
    preds_argmax = np.array([[np.argmax(each_seq) for each_seq in each_batch] for each_batch in preds_reshape])
    x_char,y_char,preds_char = [_tochar_vec(i) for i in [x,y,preds_argmax]]
    if verbose:
        print(f"""
        >>>preds: {preds.shape}
           |_reshape ==> {preds_reshape.shape}
             |_argmax ==> {preds_argmax.shape}
        """)

        print(f">>>x:{x.shape}\n",x,"\n",x_char)
        print(f">>>y:{y.shape}\n",y,"\n",y_char)
        print(f">>>preds_argmax:{preds_argmax.shape}\n",preds_argmax,"\n",preds_char)
    # 这样写也是为了防止\n在print的时候自动转义换行 | 放到数组、字典里就不会print出来换行了
    res = {"x":"".join(x_char[0]),"y":"".join(y_char[0]),"preds":"".join(preds_char[0])}
    return res
                
def print_control(cnt,info):
    if cnt % 100 == 0:
        zprint(info)
#         if cnt <= 1000:
#             # 0~1k 每100输出一次
#             zprint(info)
#         elif cnt <= 10000:
#             # 1k~10k每1k输出一次
#             if cnt % 1000 == 0:
#                 zprint(info)
#         else:
#             # 1w以后每5k输出一次
#             if cnt % 5000 == 0:
#                 zprint(info)


with model.graph.as_default():
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        counter = 0
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            et_batch_cnt = len(encoded) // (batch_size*num_steps)
            all_batch_data = get_batches(encoded, batch_size, num_steps)
            for x, y in all_batch_data:
                counter += 1
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                preds,batch_loss, new_state, _, merged_summary = sess.run([model.prediction,
                                                                     model.loss, 
                                                                     model.final_state, 
                                                                     model.optimizer,
                                                                     model.merge_summary,], 
                                                                     feed_dict=feed)
                
                # 保存进展
                model.writer.add_summary(merged_summary,counter)
                # 输出print
                info = f"epoch: {e+1}/{epochs} batch: {counter:0>3d}/{et_batch_cnt} err: {batch_loss:.4f}"
                print_control(counter,info)
                # 额外输出一个完整的字符串print
                if counter % et_batch_cnt ==0 or counter == 1:
                    text_summary_list = [tf.summary.text(k, tf.convert_to_tensor(v)) 
                                         for k,v in get_sample_char(x,y,preds).items()]
                    text_summary = tf.summary.merge(text_summary_list)
                    text_summary_ = sess.run(text_summary)
                    model.writer.add_summary(text_summary_,counter)
                # save model graph
                if (counter % save_every_n == 0):
                    _=saver.save(sess, base_model_path.format(counter, lstm_size))

        _=saver.save(sess, base_model_path.format(counter, lstm_size))


# In[39]:


len(tf.train.get_checkpoint_state("./tmp/lstm_anna").all_model_checkpoint_paths)
tf.train.get_checkpoint_state("./tmp/lstm_anna")


# ## 生成
# - 需要指定`n_samples`：需要生成多长的字符串
# - 将输入的单词转换为单个字符组成的list
# - 从第一个字符开始输入CharRNN
# - 从预测结果中选取前top_n个最可能的字符，按预测结果提供的各个字符的概率进行np.random.choice
#  - `pick_top_n`里添加了`copy()`方法，避免直接更改参数

# In[40]:


def pick_top_n(preds_, vocab_size, top_n=5, random=False):
    """
    从预测结果中选取前top_n个最可能的字符，按预测结果提供的各个字符的概率进行np.random.choice
    
    preds_: 预测结果
    vocab_size
    top_n
    """
    preds = preds_.copy()  # 避免改变原preds
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符 / 或者取概率最大的字符
    c = np.random.choice(vocab_size, 1, p=p)[0] if random else np.argmax(preds)
    return c

def sample(checkpoint, n_samples, lstm_size,num_layers, vocab_size, prime="The ", random=False):
    """
    生成新文本
    
    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    print(f">>> samples: {samples}")
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(len(char2idx), batch_size=1, num_steps=len(prime),
                    lstm_size=lstm_size, num_layers=num_layers, 
                    learning_rate=learning_rate)
    with model.graph.as_default():
        saver = tf.train.Saver()

        with tf.Session(config=sess_conf) as sess:
            # 加载模型参数，恢复训练
            saver.restore(sess, checkpoint)
            feed = {model.inputs: np.array([char2idx[c] for c in prime]),
                    model.keep_prob: 1.,}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                        feed_dict=feed,
                                        options=run_opt)
            top5_prob=preds[0][np.argsort(preds[0])[-5:]]
            top5_idx = np.argsort(preds[0])[-5:]
            print(f">>> 对整个prime: {prime} 的预测结果  [shape]:{preds.shape}")
            print(f"    top5是:{top5_prob}<==>{[idx2char[i] for i in top5_idx]}")
            next_char = pick_top_n(preds, vocab_size, random=random)
            print(f"    如果此时选取topN生成字符(是否随机:{random})，会是: [idx]:'{next_char}' [char]:'{idx2char[next_char]}'")
            
            
            # 添加字符到samples中
            samples.append(idx2char[c])
            
            inp = np.array([[c]])
            # 不断生成字符，直到达到指定数目
            for _ in range(n_samples):
                feed = {model.inputs: [[c]],
                        model.keep_prob: 1.}
                preds, new_state = sess.run([model.prediction, model.final_state], 
                                            feed_dict=feed,
                                            options=run_opt)

                c = pick_top_n(preds, vocab_size, random=random)
                samples.append(idx2char[c])

    return ''.join(samples)


# In[44]:


tf.reset_default_graph()
ckpt = tf.train.latest_checkpoint('./tmp/lstm_anna')
sample(ckpt,n_samples=2000,lstm_size=lstm_size,num_layers=num_layers,vocab_size=len(char2idx),prime="Far", random=True)


# # CharRNN 内部细节的测试

# In[14]:


with open(result_set_fp+".pickle","rb+") as frb:
    char2idx = pickle.load(frb)

encoded = np.load(coded_article_fp)


# In[15]:


def get_batches_as_iter(encoded, batch_size, time_steps, verbose=False):
    chunk_len = batch_size*time_steps 
    n_chunk = int(len(encoded)/chunk_len)
    arr = encoded[:chunk_len*n_chunk]  # 截取整数倍的batch_size
    arr = arr.reshape((batch_size,-1))

    for n in range(0, arr.shape[1], time_steps):
        x = arr[:, n:n+time_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:,1:], y[:, 0] 
        yield x, y


# In[45]:


time_steps = 10
lstm_layers = [256]*2
lstm_size = lstm_layers[0]
num_classes = len(char2idx)
default_BS = 20
default_x,default_y=list(itertools.islice(get_batches_as_iter(encoded, batch_size=default_BS, time_steps=time_steps),1))[0]
print(f">>> default_BS: {default_BS}")
print(f">>> default_x: {default_x.shape}\n",default_x[:3,:10])
print(f">>> default_y: {default_y.shape}\n",default_y[:3,:10])


# In[68]:


tf.reset_default_graph()
# placeholder
inpBS = tf.placeholder(tf.int32, [], name="batch_size")
inpX = tf.placeholder(tf.int32, shape=(None, time_steps), name="inpX")
inpY = tf.placeholder(tf.int32, shape=(None), name="inpY")
X = tf.one_hot(inpX, depth=len(char2idx))
Y = tf.one_hot(inpY, depth=len(char2idx))
# LSTM 构建
lstm_cell_list = []
for nodes_size in lstm_layers:
    lstm = tf.contrib.rnn.BasicLSTMCell(nodes_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.8)
    lstm_cell_list.append(lstm_dropout)
mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cell_list)
initial_state = mlstm_cell.zero_state(inpBS, tf.float32)
lstm_output, lstm_final_state = tf.nn.dynamic_rnn(mlstm_cell, X, initial_state = initial_state)

# formt output
seq_output = tf.concat(lstm_output, axis=1) 
softmax_x = tf.reshape(seq_output, [-1, lstm_size])
softmax_w = tf.Variable(tf.truncated_normal([lstm_size, num_classes], stddev=0.1))
softmax_b = tf.Variable(tf.zeros(num_classes))
logits = tf.matmul(softmax_x, softmax_w) + softmax_b
pred = tf.nn.softmax(logits, name='predictions')

# 计算loss
y_reshaped = tf.reshape(Y, [-1, num_classes])
loss_ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
loss = tf.reduce_mean(loss_ce)

# optimize
tvars = tf.trainable_variables()
grad_clip = 5
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
train_op = tf.train.AdamOptimizer(0.01)
optimizer = train_op.apply_gradients(zip(grads, tvars))

default_feed = {inpBS:default_BS, inpX:default_x, inpY:default_y}
with tf.Session(config=sess_conf) as sess:
    sess.run(tf.global_variables_initializer())
    inpX_,X_,inpY_,Y_,is_,lo_,lfs_each_layer = sess.run([inpX,X,inpY,Y,initial_state,lstm_output,lstm_final_state], feed_dict=default_feed)
    seq_output_,sf_x,sf_w,sf_b = sess.run([seq_output,softmax_x,softmax_w,softmax_b], feed_dict=default_feed)
    logits_,pred_,y_reshaped_ = sess.run([logits,pred,y_reshaped], feed_dict=default_feed)
    loss_ce_,loss_,_ = sess.run([loss_ce,loss,optimizer],feed_dict=default_feed)
    print(f"""\n>>> 流程如下
    inpX: {inpX_.shape}
    +onehot=> X: {X_.shape}
    +mlstm=> lstm_output: {lo_.shape}
    +reshape=> softmax_x: {sf_x.shape}
    +softmax(just matmul)=> logits: {logits_.shape}
    
    inpY: {inpY_.shape}
    +onehot=> Y: {Y_.shape}
    +reshape=> y_reshaped: {y_reshaped_.shape}
    
    CE(logits,y_reshaped): {loss_ce_.shape}
    +reduce_mean=> loss: {loss_.shape},scalar:{loss_:.4f}
    """
    )
    print(f">>> X_: {X_.shape}\n")
    print(f">>> Y_: {Y_.shape}\n")
    print(">>> lstm_final_state:")
    for idx,lfs in enumerate(lfs_each_layer):
        print(f"    >>> [layer]:{idx} [c_state:]: {lfs.c.shape}\n")
        print(f"    >>> [layer]:{idx} [h_state:]: {lfs.h.shape}\n")
    print(f">>> lstm_output: {lo_.shape}\n")
    print(f">>> seq_output_: {seq_output_.shape}\n")
    print("seq_output 的确没有起到作用,tf中对一个tensor使用concat什么都不会改变，一般是对一个内部元素是tensor的list做concat")
    print(f">>> sf_x: {sf_x.shape} sf_w: {sf_w.shape} sf_b: {sf_b.shape}")
    print(f">>> logits_: {logits_.shape} pred_: {pred_.shape}")
    
    
    


# In[ ]:




