#!/usr/bin/env python
# coding: utf-8

# 文本生成
# 
# 参照 [官网tutorials](https://www.tensorflow.org/tutorials/text/text_generation#%E4%B8%8B%E8%BD%BD%E8%8E%8E%E5%A3%AB%E6%AF%94%E4%BA%9A%E6%95%B0%E6%8D%AE%E9%9B%86)

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter,deque
import itertools
import os


# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


tf.enable_eager_execution()


# In[3]:


# 莎士比亚数据集
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# 读取并为 py2 compat 解码
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

# 创建从非重复字符到索引的映射
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
# 全文映射
text_as_int = np.array([char2idx[c] for c in text])


# In[4]:


# 文本长度是指文本中的字符个数
print ('>>> Length of text: {} characters'.format(len(text)))

# 文本中的非重复字符
print ('>>> vocab size: {} unique characters'.format(len(vocab)))

# 看一看文本中的前 250 个字符
print(">>> show head 250 characters\n",text[:250])


# 向量化文本

# In[6]:


print(f">>> text_as_int: shape {text_as_int.shape}\n",text_as_int)
print(">>> top10 of char2idx:")
for idx,(k,v) in enumerate(char2idx.items()):
    if idx <= 10:
        print(f"{repr(k):4s}:'{v:2d}'")
print(">>> top10 of idx2char:")
for i in range(10):
    print(f"{i} : {repr(idx2char[i])}")

# 显示文本首 13 个字符的整数映射
print(">>> 显示文本首 13 个字符的整数映射")
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


# 给定一个字符或者一个字符序列，下一个最可能出现的字符是什么？
# > 将文本拆分为长度为 seq_length+1 的文本块。例如，假设 seq_length 为 4 而且文本为 “Hello”， 那么输入序列将为 “Hell”，目标序列将为 “ello”。

# # 直接拷贝手册

# ## 数据

# In[7]:


# 设定每个输入句子长度的最大值
seq_length = 100
examples_per_epoch = len(text)//seq_length

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# 创建训练样本 / 目标
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
text_as_int = np.array([char2idx[c] for c in text])
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences_dataset = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences_dataset.map(split_input_target)


# 批大小
BATCH_SIZE = 64
# 设定缓冲区大小，以重新排列数据集
# （TF 数据被设计为可以处理可能是无限的序列，
# 所以它不会试图在内存中重新排列整个序列。相反，
# 它维持一个缓冲区，在缓冲区重新排列元素。） 
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset


# ## 模型

# 注意到这里`tf.keras.layers.Embedding`的参数`batch_input_shape=[batch_size, None]`，实际是指定了`batch_size`而省略了`seq_length`，也就是模型支持任意长度的句子，后面在恢复模型时使用`model.build(tf.TensorShape([1, None]))`也是只把`batch_size`指定为1，支持任意长度的句子

# In[5]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
      ])
    return model


# In[8]:


embedding_dim = 256
rnn_units = 1024
model = build_model(vocab_size = len(vocab),embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)
model.summary()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)


# ## 训练

# In[7]:


# 检查点保存至的目录
checkpoint_dir = './tmp/NLG_ckpt'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=40
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# ## 生成文本

# 恢复模型

# In[9]:


model.layers[1].states


# In[6]:


vocab_size=65
embedding_dim=256
rnn_units=1024

checkpoint_dir = './tmp/NLG_ckpt'
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
print(f">>> latest ckpt: '{latest_ckpt}'")

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(latest_ckpt)

model.build(tf.TensorShape([1, None]))
model.summary()


# In[7]:


k_sess = tf.keras.backend.get_session()
k_sess


# In[399]:


def pick_from_top_n(preds_, top_n=None, random=False, verbose=False):
    if top_n is None:
        top_n = len(preds_)
    preds = preds_.copy()  # 避免改变原preds
    p = np.squeeze(preds)
    # 小于0的都置为0
    p = np.where(p>=0, p, 0)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = np.exp(p-max(p)) / sum(np.exp(p-max(p)))
    # 输出top_N的非0概率分布
    if verbose:
        # 排序并把索引(idx2char)和值(prob)zip到一起
        prob_idx = np.stack([np.sort(p), np.argsort(p)],axis=1)[::-1]
        # 去掉0
        prob_idx = prob_idx[prob_idx[:,0]>0]
        print(prob_idx[:5])
    # 随机选取一个字符 / 或者取概率最大的字符
    c = np.random.choice(len(preds_), 1, p=p)[0] if random else np.argmax(preds)
    return c

def generate_text(model, start_string,num_generate = 20,  temperature = 1.0, random=True):
    if not random:
        print("[WARN] without random.choice, 'temperature' will not take effect")
    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]
    input_eval = np.expand_dims(input_eval, 0)

    # 空字符串用于存储结果
    text_generated = []
    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过试验以找到最好的设定


    # 这里批大小为 1
    model.reset_states()
    for i in tqdm(range(num_generate)):
        predictions = model.predict(input_eval)
        # 删除批次的维度
        predictions = np.squeeze(predictions, 0)
        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = pick_from_top_n(predictions[-1],top_n=None,random=random,verbose=False)
        
        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = np.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))


# In[404]:


np.set_printoptions(suppress=True)
# print(generate_text(model, start_string=u"ROME", num_generate=300, temperature=0.7))
print(generate_text(model, start_string=u"ROME", num_generate=100, temperature=0.1, random=True))
print("~"*20)
print(generate_text(model, start_string=u"ROME", num_generate=100, temperature=1.0, random=False))


# In[34]:


# 评估步骤（用学习过的模型生成文本）
def generate_text_custom(model, start_string, temperature=1.0, num_generate=1000, verbose=True):

    # 将起始字符串转换为数字（向量化）
    input_eval_raw = [char2idx[s] for s in start_string]  # (3,)
    input_eval = tf.expand_dims(input_eval_raw, 0)  # (1,3)
    if verbose:
        print(f">>> 输入是: '{start_string}'")
        print(f">>> +向量化: {input_eval_raw}")
        print(f">>> +维度校正: {input_eval}")
    # 空字符串用于存储结果
    text_generated = []

    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过试验以找到最好的设定
    k_sess = tf.keras.backend.get_session()
    # 这里批大小为 1
    model.reset_states()  # 清掉stateful layer的隐状态
    for i in tqdm(range(num_generate),desc="predict next char"):
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)

        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_seq = tf.random.categorical(predictions, num_samples=1)
        predicted_seq = k_sess.run(predicted_seq)
        # 最后一个字母
        predicted_id = predicted_seq[-1,0]
        text_generated.append(idx2char[predicted_id])
        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)
        if verbose:
            print(f">>> 模型prediction结果shape: {model(input_eval).shape} --squeeze at 0--> {predictions.shape}")
            print(f"    对应输入的预测字符串是: {predicted_seq.ravel()}")
            print(f"    +decode: {repr(''.join([idx2char[i] for i in predicted_seq.ravel()]))}")
            print(f"    +取最后一个字母: '{predicted_id}'")

    return (start_string + ''.join(text_generated))


# In[48]:


res = generate_text_custom(model, start_string=u"ROMEO: ", temperature=1.0, num_generate=10,verbose=False)
print(f">>> 最后输出的结果是: {repr(res)}")
print(res)


# # 预览

# In[111]:


iterator=dataset.make_one_shot_iterator()
with tf.Session() as sess:
    input_example, target_example = sess.run(iterator.get_next())
    print (f'>>> Input data: {input_example.shape} 演示的是字符索引映射回字符并join成字符串\n')
    np.array(["".join(i) for i in idx2char[input_example]])
    print (f'>>> Target data: {target_example.shape} 演示的是字符索引映射回字符并join成字符串\n')
    np.array(["".join(i) for i in idx2char[target_example]])
    print(f">>> input: {input_example.shape}\n",input_example,f"\n>>> output: {target_example.shape}\n",target_example)
    

print(">>> 模拟训练过程中的输入与label (演示每个batch的前五个)")
for i, (input_idx, target_idx) in enumerate(zip(input_example[:, :5], target_example[:, :5])):
    res = np.transpose(np.stack([idx2char[input_idx],np.full(idx2char[input_idx].shape, "--RNN->"),idx2char[target_idx]]))
    print(f"  Step-{i} expect:\n",res)


# In[82]:


# 词集的长度
vocab_size = len(vocab)

# 嵌入的维度
embedding_dim = 256

# RNN 的单元数量
rnn_units = 1024

model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
      ])
model.summary()
# 这个loss重点是把 from_logits=True  默认是False
def loss(labels, logits):
      return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer='adam', loss=loss,metrics=['acc'])
example_batch_predictions = model(input_example)
example_batch_predictions.shape


# In[67]:


ckpt_dir = "./tmp/NLG_ckpt"
ckpt_path = os.path.join(ckpt_dir, "ckpt_{epoch}")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True)


# In[110]:


dataset


# <font style='color:red'> 这里获取session必须在模型搭建完之后 </font>
# 否则会报错:
# > Error while reading resource variable xxx/xxx from Container: localhost

# 看看随机初始化的模型结果

# In[61]:


def pick_top_n(preds_, top_n=None, random=False):
    if top_n is None:
        top_n = len(preds_)
    preds = preds_.copy()  # 避免改变原preds
    p = np.squeeze(preds)
    # 小于0的都置为0
    p = np.where(p>=0, p, 0)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符 / 或者取概率最大的字符
    c = np.random.choice(len(preds_), 1, p=p)[0] if random else np.argmax(preds)
    return c
k_sess = tf.keras.backend.get_session()
pred = k_sess.run(model(input_example))
chose = np.array([[pick_top_n(char_probs) for char_probs in sequences] for sequences in pred ])
print(">>> 模型随机初始化时的predict结果 (只取第一句看效果)")
f"输入：[idx2char]:'{''.join(idx2char[input_example][0]):s}' [idx]:'{input_example[0]}'"
f"目标：[idx2char]:'{''.join(idx2char[target_example][0]):s}' [idx]:'{target_example[0]}'"
f"从输出的概率分布取top：[idx2char]:'{''.join(idx2char[chose][0]):s}' [idx]:'{chose[0]}'"
f"直接输出：[idx2char]:"
pred[0]
f"损失：[CE-loss]:"
k_sess.run(tf.keras.losses.sparse_categorical_crossentropy(target_example[0], pred[0], from_logits=True))

print(">>> 下面是详细")
print(f"输入: {input_example.shape}")
idx2char[input_example]
print(f"标注: {target_example.shape}")
idx2char[target_example]
print(f"模型预测: {pred.shape} --pick--> {chose.shape}")
idx2char[chose]
loss = k_sess.run(tf.keras.losses.sparse_categorical_crossentropy(target_example, pred, from_logits=True))
print(f"CE损失: [shape]:{loss.shape}\n",loss)


# # 问题

# 一个错误的示例 | 输入都用np转成arr

# In[ ]:


print("【实际不会用这种输入方式】如果输入是list型的，取的是第0个不管后面的")
model.reset_states()
np.squeeze(model.predict([np.array([1,2])]),1)[:,:5]
model.reset_states()
np.squeeze(model.predict([np.array([1])]),1)[:,:5]
np.squeeze(model.predict([np.array([2])]),1)[:,:5]


# <font style="color:red"> 
# 为什么？？
# 
# 预测`arr([[1,2]])`再预测`arr([[3,4]])` 
# - 等价于直接预测 `arr([[1,2],[3,4]])` ✅
# - 不等价与直接预测 `arr([[1,2,3,4]])` ❎
# </font>

# In[118]:


print("先预测arr([[1,2]]) 在预测arr([[3,4]]) ")
model.reset_states()
p1 = model.predict(np.array([[1,2]]))
p2 = model.predict(np.array([[3,4]]))
print(f">>> arr([[1,2]])的shape:{np.array([[1,2]]).shape} --输出->{p1.shape}")
np.squeeze(p1,0)[:,:5]
print(f">>> arr([[3,4]])的shape:{np.array([[3,4]]).shape} --输出->{p2.shape}")
np.squeeze(p2,0)[:,:5]

print("直接预测arr([[1,2],[3,4]])")
model.reset_states()
p1_2 = model.predict(np.array([[1,2],[3,4]]))
print(f">>> arr([[1,2],[,34]])的shape:{np.array([[1,2],[3,4]]).shape} --输出->{p1_2.shape}")
p1_2[:,:,:5]

print("直接预测arr([[1,2,3,4]])")
model.reset_states()
p12=model.predict(np.array([[1,2,3,4]]))
print(f">>> arr([[1,2,3,4]])的shape:{np.array([[1,2,3,4]]).shape} --输出->{p12.shape}")
np.squeeze(p12,0)[:,:5]


# 计算乘法

# In[ ]:


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
      ])
    return model


# In[ ]:





# In[ ]:




