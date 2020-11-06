#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[3]:


import pandas as pd


# In[8]:


df_chunks = pd.read_csv("/home/zhoutong/uidCnt.csv")
df_chunks.sort_values(by="count", ascending=False, inplace=True)
df_chunks


# DBSCAN做异常检测

# In[50]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=10, min_samples=20)
sample = np.array(df_chunks.query("count>1")['count']).reshape(-1,1)
_ = dbscan.fit_predict(sample)
# 对于DBSCAN来说，两个最重要的参数就是eps，和min_samples。当然这两个值不是随便定义的，这个在下文再说
np.unique(dbscan.labels_,return_counts=True)


# In[51]:


[i for i in np.concatenate([sample, dbscan.labels_.reshape(-1,1)],axis=1) if i[1] in [-1,0,1,2,3]]

np.unique(dbscan.labels_,return_counts=True)
dbscan.components_.shape


# 孤立森林做异常检测

# In[29]:


from sklearn.ensemble import IsolationForest
import pandas as pd

clf = IsolationForest(behaviour='new',max_samples=100, random_state=42)
table = np.array(df_chunks['count']).reshape(-1,1)
clf.fit(table)
pred = clf.predict(table)
predDF=pd.DataFrame(np.concatenate([table,pred.reshape(-1,1)], axis=1), columns=['count','isolation'])
predDF
predDF.query("isolation==-1")
predDF.query("isolation==1")


# In[ ]:





# In[ ]:





# In[ ]:




