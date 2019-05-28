#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool


# In[2]:


from zac_pyutils import ExqUtils
import pandas as pd


# In[ ]:


fileIter = ExqUtils.load_file_as_iter("./data/nlp/sample_data.txt")
def func(iter_list): return [i.split("\t") for i in iter_list]
ExqUtils.map_on_iter(fileIter,func,chunk_size=5)


# In[9]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[29]:


df_chunkList = pd.read_csv("./data/nlp/sample_data.txt", 
            delimiter="\t",
            names=['id','label','weight','feature'],
            chunksize=5,
#             iterator=False,
           )

def my_test(a,b): return str(a)+"\t"+str(b)

for chunk in df_chunkList:
    print("chunk: \n")
    chunk['feature'] = chunk['feature'].map(lambda x: x.split(","))
    chunk['id+label'] = chunk.apply(lambda row: my_test(row['id'],row['label']),axis=1)
    chunk


# In[ ]:





# In[ ]:





# In[ ]:




