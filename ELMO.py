#!/usr/bin/env python
# coding: utf-8

# In[6]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[16]:


import numpy as np
from zac_pyutils import ExqUtils
from sklearn import metrics


# # PyTorch AllenNLP
# AllenNLP 的 pyTorch 版本需要配置文件，暂时没有搞清楚配置文件怎么写

# In[1]:


from allennlp.modules.elmo import Elmo, batch_to_ids


# In[2]:


options_file = "options.json"  # 配置文件地址 
weight_file = "weights.hdf5" # 权重文件地址
# 这里的1表示产生一组线性加权的词向量。
# 如果改成2 即产生两组不同的线性加权的词向量。
elmo = Elmo(options_file, weight_file, 1, dropout=0)
# use batch_to_ids to convert sentences to character ids
sentence_lists = ["I have a dog", "How are you , today is Monday","I am fine thanks"]
character_ids = batch_to_ids(sentences_lists)
embeddings = elmo(character_ids)['elmo_representations']


# # Tensorflow_hub AllenNLP
# - [Introduction to TensorflowHub: Simple Text Embedding(Using ELMo)](https://medium.com/@joeyism/embedding-with-tensorflow-hub-in-a-simple-way-using-elmo-d1bfe0ada45c)
# - [Tensorflow+Keras做ELMo（中文翻译）](https://ai.yanxishe.com/page/TextTranslation/1597)

# In[13]:


import tensorflow as tf
import tensorflow_hub as hub
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) # trainable=True 是指使用'elmo'模式的向量结果时，ELMo的线性加和权重是可以训练的


# ## 参考自[TensorflowHub的官网](https://tfhub.dev/google/elmo/2)
# 
# ### 输入
# #### 参数
# - `signature`
#     - default | 默认是分句子输入，batch_size等于句子的个数
#         > the module takes untokenized sentences as input. The input tensor is a string tensor with shape [batch_size]. The module tokenizes each string by splitting on spaces.
#     - tokens  | 使用分过词的句子作为输入
#         > With the tokens signature, the module takes tokenized sentences as input. The input tensor is a string tensor with shape [batch_size, max_length] and an int32 tensor with shape [batch_size] corresponding to the sentence length. The length input is necessary to exclude padding in the case of sentences with varying length.
#     - 上述两种方式的结果是完全一致的
# - `as_dict`
#     - 不提供参数 `as_dict`时（默认为 `False`），会使用`default`。即 `elmo(tokens, as_dict=True)['default']` = `elmo(tokens)`
# - 有句子sen0,sen1,sen2；
#   - 分开输入：分三次输入（使用'default'）得到三个句子向量emb0,emb1,emb2；
#   - 一次输入：作为一个list [sen0,sen1,sen2] 一次性输入，得到一个emb_list；
#   - 根据测试结果来看，emb_lit[0] 和 emb0 余弦相似度 0.99999+

# In[11]:


# 使用 signature = default， 输入是句子
sentences = ["the cat is on the mat", "dogs are in the fog"]
emb_opt_sentence = elmo(inputs=sentences, as_dict=True)

# 使用 signature = tokens, 输入是字典，包含 tokens 和 sequence_len 注意需要padding
def padding(tokens_inp,pad=""):
    pad_len = max([len(i) for i in tokens_inp])
    return [(i+[pad]*pad_len)[:pad_len] for i in tokens_inp]

tokens_list = ["the cat is on the mat".split(" "),"dogs are in the fog".split(" ")]
tokens_list = padding(tokens_list,"__PAD__")
tokens_len = [len(i) for i in tokens_list]
input_dict = {
    'tokens': tokens_list,
    'sequence_len' : tokens_len
}
input_dict
emb_opt_tokens = elmo(inputs=input_dict, signature='tokens', as_dict=True)


# ### 输出
# > The output dictionary contains:
# >- <u>word_emb</u>: 
# >    - the character-based word representations 
# >    - general embedding 与上下文无关，ELMo使用的是charater-based CNN + Highway
# >    - `elmo(tokens, as_dict=True)['word_emb']`
# >    - *[batch_size, max_length, 512]*
# >- <u>lstm_outputs1</u>: 
# >    - the first LSTM hidden state 
# >    - 双向LSTM中的一个
# >    - `elmo(tokens, as_dict=True)['lstm_outputs1']`
# >    - *[batch_size, max_length, 1024].*
# >- <u>lstm_outputs2</u>: t
# >    - the second LSTM hidden state 
# >    - 双向LSTM中的一个
# >    - `elmo(tokens, as_dict=True)['lstm_outputs2']`
# >    - *[batch_size, max_length, 1024].*
# >- <u>elmo</u>: 
# >    - the weighted sum of the 3 layers, where the weights are trainable. 
# >    - 上述三个（word+2*lstm) emb的线性加权组合，针对不同任务时这里的权重（维度为3）可以进行训练，即设置的Trainable
# >    - `elmo(tokens, as_dict=True)['elmo']`
# >    - *[batch_size, max_length, 1024]*
# >- <u>default</u>: 
# >    - a fixed mean-pooling of all contextualized word representations 
# >    - 即平均后的句子向量
# >    - `elmo(tokens, as_dict=True)['default']`
# >    - *[batch_size, 1024].*
# 
# 当输入采用不同方式tokens/sentences时，只有word_emb能确保五种输出结果是一致的；
# 
# 不一致的原因实际上是因为句子分词后长度不同，需要用空串padding；
# 
# 如果句子分词后长度都一直，则五种输出结果都一致。

# In[12]:


with tf.Session() as sess:
    for output in ['word_emb','lstm_outputs1','lstm_outputs2','elmo','default']:
        sess.run(tf.global_variables_initializer())
        emb = sess.run(emb_opt_sentence[output])
        emb2 = sess.run(emb_opt_tokens[output])
        print(f"signature不同的输入方式(tokens,sentences)，'{output}' 结果是一致的: ",np.all(emb == emb2))


# In[148]:


# 计算好emb
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    emb_tokens = sess.run(emb_opt_tokens['elmo'])
    emb_sentence = sess.run(emb_opt_sentence['elmo'])


# In[167]:


print("sentences: ", sentences)
for sen_idx in [0,1]:
    print("\n>>> 使用tokens_list: ", tokens_list[sen_idx])
    consistent = np.all(emb_tokens[sen_idx] == emb_sentence[sen_idx])
    print("两种输入方式下 elmo 的一致性：",consistent)
    if not consistent:
        print("不一致时，tokens和sentence两种方式的tensor值示例：")
        emb_tokens[sen_idx]
        emb_sentence[sen_idx]


# # 验证

# ## 句子相似度度量

# In[17]:


pad_len = 13
demoSentence_total = ["Modi was appointed Chief Minister of Gujarat in 2001",
                      "Narendra Modi was appointed Chief Minister of Gujarat in 2001",
                      "Obama was appointed Chief Minister of Gujarat in 2001"]
demoSentence_total = [" ".join(ExqUtils.padding(sen.split(" "),pad_len)) for sen in demoSentence_total]
demoSentence_total

with tf.Session() as sess:
    emb_opt = elmo(inputs=sentences, as_dict=True)
    sess.run(tf.global_variables_initializer())
    emb_demo_opt = tf.reshape(elmo(inputs=demoSentence_total, as_dict=True)['default'],[3,1,1024])
    emb_demo_total = sess.run(emb_demo_opt)


# In[ ]:


emb_demo_total = emb_demo_total.reshape(3,1,1024)
metrics.pairwise.cosine_similarity(emb_demo_total[0],emb_demo_total[1])
metrics.pairwise.cosine_similarity(emb_demo_total[0],emb_demo_total[2])


# In[ ]:




