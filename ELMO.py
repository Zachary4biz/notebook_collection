#!/usr/bin/env python
# coding: utf-8

# In[61]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[120]:


import numpy as np


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

# In[6]:


import tensorflow as tf
import tensorflow_hub as hub
elmo = hub.Module("https://tfhub.dev/google/elmo/2")


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

# In[131]:


# 使用 signature = default， 输入是句子
sentences = ["the cat is on the mat", "dogs are in the fog"]
emb_opt_sentence = elmo(inputs=sentences, as_dict=True)

# 使用 signature = tokens, 输入是字典，包含 tokens 和 sequence_len 注意需要padding
def padding(tokens_inp):
    pad_len = max([len(i) for i in tokens_inp])
    return [(i+[""]*pad_len)[:pad_len] for i in tokens_inp]

tokens_list = ["the cat is on the mat".split(" "),"dogs are in the fog".split(" ")]
tokens_list = padding(tokens_list)
tokens_len = [len(i) for i in tokens_list]
input_dict = {
    'tokens': tokens_list,
    'sequence_len' : tokens_len
}
emb_opt_tokens = elmo(inputs=input_dict, signature='tokens', as_dict=True)


# ### 输出
# > The output dictionary contains:
# >- <u>word_emb</u>: 
# >    - the character-based word representations 
# >    - `elmo(tokens, as_dict=True)['word_emb']`
# >    - *[batch_size, max_length, 512]*
# >- <u>lstm_outputs1</u>: 
# >    - the first LSTM hidden state 
# >    - `elmo(tokens, as_dict=True)['lstm_outputs1']`
# >    - *[batch_size, max_length, 1024].*
# >- <u>lstm_outputs2</u>: t
# >    - he second LSTM hidden state 
# >    - `elmo(tokens, as_dict=True)['lstm_outputs2']`
# >    - *[batch_size, max_length, 1024].*
# >- <u>elmo</u>: 
# >    - the weighted sum of the 3 layers, where the weights are trainable. 
# >    - `elmo(tokens, as_dict=True)['elmo']`
# >    - *[batch_size, max_length, 1024]*
# >- <u>default</u>: 
# >    - a fixed mean-pooling of all contextualized word representations 
# >    - `elmo(tokens, as_dict=True)['default']`
# >    - *[batch_size, 1024].*
# 
# 当输入采用不同方式tokens/sentences时，只有word_emb能确保五种输出结果是一致的；
# 
# 不一致的原因实际上是因为句子分词后长度不同，需要用空串padding；
# 
# 如果句子分词后长度都一直，则五种输出结果都一致。

# In[132]:


with tf.Session() as sess:
    for output in ['word_emb','lstm_outputs1','lstm_outputs2','elmo','default']:
        sess.run(tf.global_variables_initializer())
        emb = sess.run(emb_opt_sentence[output])
        emb2 = sess.run(emb_opt_tokens[output])
        print(f"signature不同的输入方式(tokens,sentences)，'{output}' 结果是一致的: ",np.all(emb == emb2))


# In[148]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    emb_tokens = sess.run(emb_opt_tokens['elmo'])
    emb_sentence = sess.run(emb_opt_sentence['elmo'])


# In[153]:


for sen_idx in [0,1]:
    sentences[sen_idx]
    tokens_list[sen_idx]
    consistent = np.all(emb_tokens[sen_idx] == emb_sentence[sen_idx])
    print("elmo的一致性：",consistent)
    if not consistent:
        emb_tokens[sen_idx]
        emb_sentence[sen_idx]


# In[ ]:




