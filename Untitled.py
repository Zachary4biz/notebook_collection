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
from collections import Counter


# In[2]:


import pandas as pd


# In[9]:


df_chunks = pd.read_csv("/home/zhoutong/client_profile_20190924103911.csv", encoding='gbk',chunksize=10000)


# In[ ]:





# In[16]:


set(df_chunks.get_chunk()['user_pin'])


# In[ ]:




