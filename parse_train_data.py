#!/usr/bin/env python
# coding: utf-8

# 涂博的人脸样本数据，根据对应的json文件获得各种属性的分类样本

# # Prepare

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


import os
import json
import cv2
from matplotlib import pyplot as plt


# # 找出所有json文件路径，保存下来

# In[4]:


json_fp = "/home/zhoutong/facedata/all_json_file_path.txt"


# In[19]:


# 找出所有json文件路径，保存下来
def write2file_useful_json(output_fp="/home/zhoutong/facedata/all_json_file_path.txt"):
    base_dir = "/home/zhoutong/facedata/CASIA-maxpy-clean"
    sub_dir_list = [os.path.join(base_dir, i) for i in os.listdir(base_dir)]
    res_path_list = []
    for sub_dir in sub_dir_list:
        if os.path.isdir(sub_dir):
            res_path_list.extend([os.path.join(sub_dir, i) for i in os.listdir(sub_dir) if ".json" in i])
    with open(output_fp, "w") as f:
        for p in res_path_list:
            f.writelines(p + "\n")
    return None


# In[ ]:


# 找出所有json文件路径，保存下来
write2file_useful_json(json_fp)


# # fp_list 存有所有json文件路径

# In[5]:


with open(json_fp, "r") as f:
    fp_list = [i.strip() for i in f.readlines()]


# # 逐一打开json文件，按描述取出人脸图像

# In[135]:


def process_img(img,rect_list,expand_r=0.15):
    img_list = []
    H,W,_ = img.shape
    for rect in rect_list:
        (top,left,width,height) = (rect['top'],rect['left'],rect['width'],rect['height'])
        top = int(top - expand_r*height)
        height = int((1+expand_r*2)*height)
        left = int(left - expand_r*width)
        width = int((1+expand_r*2)*width)
        top = top if top>0 else 0
        left = left if left >0 else 0
        img_list.append(img[top:top+height,left:left+width])
    return img_list


# In[ ]:


for fp in tqdm_notebook(fp_list[:]):
    new_fp = get_new_fp(fp)
    img_fp = os.path.splitext(json_fp)[0]+".jpg"
    img = cv2.imread(img_fp)
    with open(json_fp,"r") as f:
        json_info = json.load(f)
    rect_list = [json_info['face_rectangle'] for json_info in content['faces']]


# In[ ]:


def process_img_old(img,rect_list):
    img_list = []
    for rect in rect_list:
        (top,left,width,height) = (rect['top'],rect['left'],rect['width'],rect['height'])
        img_list.append(img[top:top+height,left:left+width])
    return img_list

def prepare(json_fp):
    img_fp = os.path.splitext(json_fp)[0]+".jpg"
    with open(json_fp,"r") as f:
        content = json.load(f)
    img = cv2.imread(img_fp)
    rect_list = [i['face_rectangle'] for i in content['faces']]
    attr_list = [i['attributes']['ethnicity']['value'] for i in content['faces']]
    face_list = process_img(img,rect_list)
    return face_list,attr_list

def get_new_fp(fp="/home/zhoutong/facedata/CASIA-maxpy-clean/0000133/007.json"):
    CASIA_dir,img_collection_dir = os.path.split(os.path.dirname(fp))
    f_name = img_collection_dir+"_"+os.path.basename(fp).replace("json","jpg")
    f_dir = os.path.abspath(os.path.join(os.path.join(CASIA_dir,"../"),"prepared_data","face_img"))
    return os.path.join(f_dir,f_name)

def show(face,eth):
    _ = plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    print("ethnicity: ",eth)
    plt.show()


# In[139]:


for fp in tqdm_notebook(fp_list[0:]):
    new_fp = get_new_fp(fp)
#     print(fp)
#     print(new_fp)
    face_list,ethnicity_list = prepare(fp)
    with open("/home/zhoutong/facedata/prepared_data/enthnicity.csv","a") as f:
        for face,eth in zip(face_list,ethnicity_list):
    #         show(face,eth)
            save = cv2.imwrite(new_fp,face)
            if not save:
                print("Fail: [fp]:{} [new_fp]:{}".format(fp,new_fp))
            else:
                f.writelines(new_fp+"\t"+eth+"\n")


# In[160]:


with open("/home/zhoutong/facedata/prepared_data/enthnicity.csv","r") as f:
    content = [i.strip().split("\t") for i in f.readlines()][:10]


# # 【题外操作】人种数据分目录存好，用TF的InterceptionV3分类

# In[17]:


with open("/home/zhoutong/facedata/prepared_data/enthnicity_train_balanced.csv","r") as f:
    content = [i.strip().split("\t") for i in f.readlines()]


# In[18]:


set([i[1] for i in content])
for a,b in tqdm_notebook(content[:]):
    p = "/home/zhoutong/facedata/prepared_data/enthnicity_img_copied/{}/{}"
    p = p.format(b,os.path.split(a)[-1])
    img = cv2.imread(a)
    res = cv2.imwrite(p,img)
    if not res:
        print(a,b)
        print(p)


# # Train

# In[3]:


import numpy as np
import os
import tensorflow as tf

import mobilenet_v2

import cv2
import math
import json
import pandas as pd
import random
import datetime


# In[4]:


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMAGE_WIDTH = IMG_WIDTH
IMAGE_HEIGHT = IMG_HEIGHT
IMG_CHANNEL = 3
shuffle = True
CHECKPOINTS_DIR = '/home/zhoutong/facedata/prepared_data/ckpt/enthnicity/'

BATCH_SIZE = 20
NUM_EPOCHS = 40000000000

global modelValLoss
global totalValLoss
global valIter
valIter = 0

initialLearningRate = 0.001
learningRateDecay = 0.0000001
weightDecay = 0.0005
label_dim = 4 # {'ASIAN', 'BLACK', 'INDIA', 'WHITE'}
label_dict = {'ASIAN':[1,0,0,0], 'BLACK':[0,1,0,0], 'INDIA':[0,0,1,0], 'WHITE':[0,0,0,1]}

def load_input(fp):
    with open(fp,"r") as f:
        c = [i.strip().split("\t") for i in f.readlines()]
        random.shuffle(c)
        img_fp = [i[0] for i in c]
        label = [label_dict[i[1]] for i in c]
    return img_fp,label

def zprint(inp):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("|{}| ".format(now)+inp)
    
    
train_img_fp,train_label = load_input('/home/zhoutong/facedata/prepared_data/enthnicity_train_balanced.csv')
test_img_fp,test_label = load_input('/home/zhoutong/facedata/prepared_data/enthnicity_test.csv')


# In[5]:


test_label.count(label_dict["ASIAN"])
test_label.count(label_dict["BLACK"])
test_label.count(label_dict["INDIA"])
test_label.count(label_dict["WHITE"])


# ## TrainEpoch

# In[6]:


inputs = tf.placeholder("float32", [None, IMG_HEIGHT, IMG_WIDTH, 3], name="input_to_float")
inptrue = tf.placeholder("float32", [None, 4], name="inptrue")
global_step = tf.placeholder("int32", name="global_step")
logits, pred = mobilenet_v2.mobilenetv2_caffe(inputs, 4) # [0.134,0.34426,0.77456,0.23523], 0.78


# In[20]:


loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=inptrue,logits=logits),name='loss')
_ = tf.summary.scalar('loss', loss)
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(inptrue,0), 
                                  predictions=tf.argmax(logits,0))
_ = tf.summary.scalar('acc', acc)
learningRate = tf.train.inverse_time_decay(initialLearningRate, global_step, 1, learningRateDecay, name="ratedecay")
optimizer = tf.train.AdamOptimizer(learningRate, name="adamfull")
train_op = optimizer.minimize(loss, name="trainop")
merged = tf.summary.merge_all()


# In[21]:


def train_epoch(train_writer,merged,k,sess):
    i = 0
    global valIter
    batchCounter = 0
    random.shuffle(train_img_fp)
    while i < len(train_img_fp):
        batchTrainImages = []
        batchTrainLabels = []

        j = i
        while j < i + BATCH_SIZE and j < len(train_img_fp):
            label = train_label[j]
            trainImage = cv2.imread(train_img_fp[j])
            if trainImage is None:
                continue

            if trainImage.shape[0] != IMAGE_HEIGHT or trainImage.shape[1] != IMAGE_WIDTH:
                trainImage = cv2.resize(trainImage, (IMAGE_WIDTH, IMAGE_HEIGHT))

            trainImage = trainImage.astype("float32")
            trainImage = trainImage / 255.0
            batchTrainImages.append(trainImage)
            batchTrainLabels.append(label)
            j = j + 1

        acc_,summary,loss, _ = sess.run([acc,merged,"loss:0", "trainop"], feed_dict={
            "input_to_float:0": batchTrainImages,
            "inptrue:0": batchTrainLabels,
            "global_step:0": batchCounter,
        })
        train_writer.add_summary(summary, batchCounter+int(k*(len(train_img_fp)/BATCH_SIZE)))
        
        if batchCounter%400 == 0:
            zprint("[batch]:{} [loss]:{} [acc]:{}".format(batchCounter,loss,acc_))
        batchCounter += 1
        i = i + BATCH_SIZE


# In[22]:


def performValidation(sess, k):
    # calculate validation loss first
    i = 0
    batchCounter = 0
    totalValLoss = 0

    global modelValLoss

    while i < len(test_img_fp):
        batchValImages = []
        batchValLabels = []

        j = i
        while j < i + BATCH_SIZE and j < len(test_img_fp):
            label = test_label[j]
            valImage = cv2.imread(test_img_fp[j])
            if valImage is None:
                continue


            if valImage.shape[0] != IMAGE_HEIGHT or valImage.shape[1] != IMAGE_WIDTH:
                valImage = cv2.resize(valImage, (IMAGE_WIDTH, IMAGE_HEIGHT))

            valImage = valImage.astype("float32")
            valImage = valImage / 255.0

            batchValLabels.append(label)
            batchValImages.append(valImage)
            j = j + 1

        acc_,loss = sess.run([acc,"loss:0"], feed_dict={
            "input_to_float:0": batchValImages,
            "inptrue:0": batchValLabels,
            "global_step:0": batchCounter,
        })

        #loss = loss[0] / float(len(batchValImages))

        totalValLoss += loss
        batchCounter += 1
        i = i + BATCH_SIZE

    zprint("[Test-iteration]: {} [loss]: {} [acc]:{}".format(k, totalValLoss / float(batchCounter),acc_))
    if (k == 0):
        modelValLoss = totalValLoss
        saver.save(sess, CHECKPOINTS_DIR + "enetperson-" + str(k) + ".ckpt")
        zprint('-'*15+'Saving the model(initial)'+'-'*15)
    else:
        if (totalValLoss < modelValLoss):
            modelValLoss = totalValLoss
            saver.save(sess, CHECKPOINTS_DIR + "enetperson-" + str(k) + ".ckpt")
            zprint('-'*15+'Saving the model'+'-'*15)


# In[23]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[24]:


with tf.Session(config=config) as sess:
    saver = tf.train.Saver(max_to_keep=20)
    train_writer = tf.summary.FileWriter('/home/zhoutong/facedata/prepared_data/summaries',sess.graph)
    _,_=sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    else:
        ckpt_path = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
        print("ckpt_path: ",ckpt_path)
        if not ckpt_path is None:
            print('restore the last training', ckpt_path)
            saver.restore(sess, ckpt_path)
        else:
            print('no restore this time', ckpt_path)

    print('----------------------begin trainng...---------------------------------------')
    k = 0
    writer = tf.summary.FileWriter(CHECKPOINTS_DIR, sess.graph)
    while k < NUM_EPOCHS:
        # start training
        print( "Starting training")
        train_epoch(train_writer,merged,k,sess)
        performValidation(sess, k)
        print( "Epoch training complete ", k)
        k = k + 1


# # Predict

# In[28]:


label_dict_ = {'ASIAN':[1,0,0,0], 'BLACK':[0,1,0,0], 'INDIA':[0,0,1,0], 'WHITE':[0,0,0,1]}
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    _,_=sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    ckpt_path = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
    print("ckpt_path: ",ckpt_path)
    valImage = cv2.imread("/home/zhoutong/facedata/prepared_data/face_img/6573530_054.jpg")
    if valImage.shape[0] != IMAGE_HEIGHT or valImage.shape[1] != IMAGE_WIDTH:
        valImage = cv2.resize(valImage, (IMAGE_WIDTH, IMAGE_HEIGHT))
    valImage = valImage.astype("float32")
    valImage = valImage / 255.0
    l,p = sess.run([logits,pred],feed_dict={inputs:[valImage]})
    l
    p
    l.argmax()
    p.argmax()


# In[ ]:





# # Test

# In[115]:


with open("/home/zhoutong/facedata/CASIA-maxpy-clean/0000133/007.json","r") as f:
    json_str = json.load(f)
json_str


# In[131]:


import pandas as pd
examples = pd.read_csv("/home/zhoutong/data_train_casia.csv")
examples.head(3)
imageNames = []
for _, row in examples.iterrows():
    linedata = row
    imname, jsonname = linedata['jpg'], linedata['json']
    imageNames.append((imname, jsonname))
imageNames[0]


# In[ ]:




