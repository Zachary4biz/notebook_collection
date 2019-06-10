#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[57]:


import numpy as np
import re
from zac_pyutils import ExqUtils
from sklearn import metrics


# In[4]:


import tensorflow as tf
import tensorflow_hub as hub
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable = True)


# In[6]:


article = """Get News Updates directly on your WhatsApp. Click here to Subscribe.\n\nRanveer Singh and Deepika Padukone have been giving us major couple goals. And today, Deepika took to her Instagram to share her look for an award night.\n\n\n\n\n\nShe shared a series of photos from which can make anyone go weak in the knees. Ranveer also got awestruck after seeing the photos.DP looked absolutely stunning in a pink dress with a ruffled neck and Ranveer couldn't stop himself from commenting on the posts shared by the actress.\n\nAt the award function Deepika's father Prakash Padukone received Lifetime Achievement Award for his contribution in sports. Ranveer also attended the event to witness the moment.On the work front, Deepika will be next seen in Meghna Gulzar 's 'Chhapaak' while Ranveer's next is Kabir Khan's '83'."""
sentences = [re.sub("\\n","",i.strip()) for i in article.strip().split(".")]
sentences = [i for i in sentences if len(i)>0]
sentences


# In[74]:


pad_len = 13
demoSentence_total = ["Modi was appointed Chief Minister of Gujarat in 2001",
                      "Narendra Modi was appointed Chief Minister of Gujarat in 2001",
                      "Obama was appointed Chief Minister of Gujarat in 2001"]
demoSentence_total = [" ".join(ExqUtils.padding(sen.split(" "),pad_len)) for sen in demoSentence_total]
demoSentence_total


# In[72]:


with tf.Session() as sess:
    emb_opt = elmo(inputs=sentences, as_dict=True)
    sess.run(tf.global_variables_initializer())
#     emb_sentence = sess.run(emb_opt['elmo'])
    emb_demo_opt = tf.reshape(elmo(inputs=demoSentence_total, as_dict=True)['default'],[3,1,1024])
    emb_demo_total = sess.run(emb_demo_opt)


# In[11]:





# In[ ]:


def cos_similarity(vec1,vec2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(vec1), axis=2))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(vec2), axis=2))


# In[9]:


[len(i) for i in emb_sentence]


# In[56]:





# In[8]:


ExqUtils.padding([12,123,14],5)


# In[ ]:




