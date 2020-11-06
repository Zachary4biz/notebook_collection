#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[2]:


import numpy as np
import re
from zac_pyutils import ExqUtils
from sklearn import metrics


# In[3]:


import tensorflow as tf
import tensorflow_hub as hub
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable = True)


# In[4]:


article = """Get News Updates directly on your WhatsApp. Click here to Subscribe.\n\nRanveer Singh and Deepika Padukone have been giving us major couple goals. And today, Deepika took to her Instagram to share her look for an award night.\n\n\n\n\n\nShe shared a series of photos from which can make anyone go weak in the knees. Ranveer also got awestruck after seeing the photos.DP looked absolutely stunning in a pink dress with a ruffled neck and Ranveer couldn't stop himself from commenting on the posts shared by the actress.\n\nAt the award function Deepika's father Prakash Padukone received Lifetime Achievement Award for his contribution in sports. Ranveer also attended the event to witness the moment.On the work front, Deepika will be next seen in Meghna Gulzar 's 'Chhapaak' while Ranveer's next is Kabir Khan's '83'."""


# # Prepare

# ## split into sentences

# In[5]:


def split_to_sentences(article):
    sentences = [re.sub("\\n","",i.strip()) for i in article.strip().split(".")]
    sentences = [i for i in sentences if len(i)>0]
    return sentences


# ## pad/truncate to same length

# In[6]:


def align(sentence_list, pad_len=13):
    sentences_pad = [" ".join(ExqUtils.padding(sen.split(" "),pad_len)) for sen in sentence_list]
    return sentences_pad


# In[7]:


padded_sens = align(split_to_sentences(article),pad_len=30)


# In[21]:


ExqUtils.zprint(f"""句子: {len(padded_sens)}, 词(each sentences): {len(padded_sens[0].split(" "))}""")
padded_sens


# # TF | Graph

# In[22]:


g0 = tf.get_default_graph()
with g0.as_default():
    emb_opt = elmo(inputs=padded_sens, as_dict=True)
    word_emb = emb_opt['word_emb']
    LM_emb = tf.concat([emb_opt['lstm_outputs1'], emb_opt['lstm_outputs2']],axis=2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        emb_res = sess.run(emb_opt)
        LM_emb_res = sess.run(LM_emb)
        


# In[23]:


emb_res['lstm_outputs1'].shape
emb_res['lstm_outputs2'].shape
LM_emb_res.shape # 10个句子，每个句子30个词，每个词1024维向量


# In[ ]:


with tf.Session() as sess:
    emb_opt = elmo(inputs=sentences, as_dict=True)
    sess.run(tf.global_variables_initializer())
#     emb_sentence = sess.run(emb_opt['elmo'])
    emb_demo_opt = tf.reshape(elmo(inputs=demoSentence_total, as_dict=True)['default'],[3,1,1024])
    emb_demo_total = sess.run(emb_demo_opt)


# In[ ]:





# In[ ]:


def cos_similarity(vec1,vec2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(vec1), axis=2))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(vec2), axis=2))


# In[ ]:


[len(i) for i in emb_sentence]


# In[ ]:





# In[ ]:


ExqUtils.padding([12,123,14],5)


# In[ ]:




