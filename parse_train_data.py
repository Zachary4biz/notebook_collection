#!/usr/bin/env python
# coding: utf-8

# In[22]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[37]:


import os
import json
import cv2
from matplotlib import pyplot as plt


# In[8]:


base_dir = "/home/zhoutong/facedata/CASIA-maxpy-clean"
sub_dir_list = [os.path.join(base_dir,i) for i in os.listdir(base_dir)]


# In[12]:


res_path_list = []
for sub_dir in sub_dir_list:
    if os.path.isdir(sub_dir):
        res_path_list.extend([os.path.join(sub_dir,i) for i in os.listdir(sub_dir) if ".json" in i])


# In[14]:


with open("/home/zhoutong/facedata/all_json_file_path.txt","w") as f:
    for i in res_path_list:
        f.writelines(i+"\n")


# In[43]:


json_path = res_path_list[0]
img_path = os.path.splitext(res_path_list[0])[0]+".jpg"
img = cv2.imread(img_path)
with open(res_path_list[0],"r") as f:
    feature = json.load(f)
# [{'face_rectangle':i['face_rectangle'],'ethnicity':i['ethnicity'],'beauty':i['beauty']} for i in feature['faces']]

rect_list = [i['face_rectangle'] for i in feature['faces']]
(top,left,width,height) = (rect_list[0]['top'],rect_list[0]['left'],rect_list[0]['width'],rect_list[0]['height'])
attribute_list = [i['attributes']['ethnicity'] for i in feature['faces']]

plt.imshow(img)
plt.show()

plt.imshow(img[top:top+height,left:left+width])
plt.show()


# In[ ]:




