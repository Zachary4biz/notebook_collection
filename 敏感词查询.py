#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[ ]:


text = ""
sensitiveWords = dict(
    1:[""],
    2:[""],
    3:[""],
#     4:[""], ...
)


# In[ ]:


class 

