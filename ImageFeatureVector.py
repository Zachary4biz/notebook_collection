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


# In[ ]:


import fasttext
import json
import itertools
from tqdm import tqdm
from zac_pyutils import ExqUtils  # from pip install
from zac_pyutils.ExqUtils import zprint


# In[ ]:




