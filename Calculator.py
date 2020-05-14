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
import itertools
import os


# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt


# In[212]:


sample_cnt = 100
x=np.random.randint(low=0,high=9,size=(sample_cnt,2))
y=x[:,0]*x[:,1]
samples = np.hstack((x,y.reshape(sample_cnt,-1)))
samples[:10]


# In[216]:


item2idx=dict([(str(i),i) for i in range(0,10)])
for item in samples[2:3]:
    for number in item:
        [item2idx[char] for char in str(number)]
            


# In[176]:


class Calc(tf.keras.Model):
    def __init__(self,total_size=100,emb_dim=3,**kwargs):
        super().__init__(name="M",**kwargs)
        self.emb = tf.keras.layers.Embedding(total_size, emb_dim)
#         self.emb = tf.keras.layers.Embedding(total_size, emb_dim)
#         self.RNN = tf.keras.layers.GRU(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform')
        self.lstm = tf.keras.layers.LSTM(16, return_sequences=False)
        self.prediction = tf.keras.layers.Dense(total_size)
        
    def call(self, inp, training=False):
        x = self.emb(inp)
        x = self.lstm(x)
        x = self.prediction(x)
        return x
    
M = Calc(total_size=10)
# M.build((1,4))
M(tf.zeros((1,10)))
M.summary()


M.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.losses.MSE, metrics=['acc'])


# In[ ]:


model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])


# In[ ]:





# In[105]:


np.random.seed(9)
vocab_size = 10 # 所有词有多少个
seq_length = 3 # 一个句子里词的个数 或者 一个样本的特征个数
emb_dim = 3     # 每个词的词向量维数 或者 一个特征的隐向量维数
batch_size = 2  # 一个输入batch大小 | inp -> emb: (batch_size, seq_length) -> (batch_size, seq_length, emb_dim)
lstm_hidden_size = 16

input_arr = np.random.randint(vocab_size, size=(batch_size, seq_length))
input_arr.shape
input_arr

"embl"
embl=tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)#, input_length=seq_length)
embl(input_arr).numpy().shape
embl(input_arr).numpy()

"densel"
densel=tf.keras.layers.Dense(units=4)
densel(embl(input_arr)).numpy().shape
densel(embl(input_arr)).numpy()
"densel kernel bias"
densel.kernel.shape
densel.bias.shape


"flattenl"
flattenl=tf.keras.layers.Flatten()
flattenl(embl(input_arr)).numpy().shape
flattenl(embl(input_arr)).numpy()

"densel(flattenl)"
densel=tf.keras.layers.Dense(units=4)
densel(flattenl(embl(input_arr))).numpy().shape
densel(flattenl(embl(input_arr))).numpy()
"densel(flattenl) kernel bias"
densel.kernel.shape
densel.bias.shape

"lstml0"
lstml0=tf.keras.layers.LSTM(lstm_hidden_size, return_sequences=True)
lstml0(embl(input_arr)).numpy().shape
lstml0(embl(input_arr)).numpy()

"lstml1"
lstml1=tf.keras.layers.LSTM(lstm_hidden_size, return_sequences=False)
lstml1(embl(input_arr)).numpy().shape
lstml1(embl(input_arr)).numpy()
# lstml1=tf.keras.layers.LSTM(1, activation='sigmoid', return_sequences=False)
# lstml0.build((batch_size))
# lstml0.count_params()

"lstml2"
lstml2=tf.keras.layers.LSTM(lstm_hidden_size, return_sequences=True, return_state=True)
output,h,c=[i.numpy() for i in lstml2(embl(input_arr))]
[i.numpy().shape for i in lstml2(embl(input_arr))]
f"lstml2|output {output.shape}"
output
f"lstml2|h {h.shape}"
h
f"lstml2|c {c.shape}"
c
f"lstml2|output[:,-1,:] {output[:,-1,:].shape}"
"output[:,-1,:] 就是 h"
output[:,-1,:]


# In[161]:


np.all(lstml1.cell.bias.numpy()==lstml1.get_weights()[2])
np.all(lstml1.cell.recurrent_kernel.numpy() == lstml1.get_weights()[1])
np.all(lstml1.cell.kernel.numpy() == lstml1.get_weights()[0])
[w.shape for w in lstml1.get_weights()]
"cell.kernel, cell.recurrent_kernel, cell.bias"
lstml1.count_params()


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

