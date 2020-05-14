#!/usr/bin/env python
# coding: utf-8

# 基于Incepiton的迁移学习，用 tf、keras、tfhub 实现
# 
# 参照 [官网tutorials](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub#download_the_headless_model)

# In[8]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys
from collections import Counter,deque
import itertools
import os
import time


# In[69]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import pandas as pd
import subprocess
import dlib
from zac_pyutils import ExqUtils


# 只能检测到"XLA_GPU"，但是不能使用，<font style="color:red">CUDA 和 cuDNN 需要升级</font>
# - 参考这个issue：https://github.com/tensorflow/tensorflow/issues/30388
# > Finally, you can get rid of this issue by uninstalling / reinstalling (tested on Ubuntu 18.04):
# >
# > Tensorflow 2.0
# >
# > CUDA 10.0
# >
# > cuDNN 7.6.4 (described as dedicated for CUDA 10.0)
# >
# > https://www.tensorflow.org/install/source#tested_build_configurations. You will get xla devices with corresponding non xla devices.

# In[ ]:


# 【注意】这里 set_log_device_placement 打开了自后后面加载模型都会有很多log
tf.config.experimental.list_physical_devices()
tf.config.experimental.list_physical_devices('GPU')
print(">>> 验证是否能在GPU上计算")
tf.debugging.set_log_device_placement(True)
# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


# tf1.x 里通过session控制

# In[3]:


sess_conf = tf.ConfigProto()
sess_conf.gpu_options.allow_growth = True  # 允许GPU渐进占用
sess_conf.allow_soft_placement = True  # 把不适合GPU的放到CPU上跑

g_graph = tf.Graph()
g_sess = tf.Session(graph=g_graph, config=sess_conf)


# ~~tf2.x 用新的方式控制~~

# In[4]:


import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.75)
tf.config.gpu.set_per_process_memory_growth(True)


# tf2.x 用新的方式控制

# In[10]:


gpus = tf.config.experimental.list_physical_devices('GPU')
gpus
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# # 图片分类

# 图片切割出人脸重新存储

# In[ ]:


import dlib
dlib_detector = dlib.get_frontal_face_detector()

def get_face_imgArr(imgArr, enlarge=0.2):
    imgArr = imgArr#.astype(np.dtype('|u1'))
    img_gray = np.array(Image.fromarray(imgArr).convert("L"))
    rect_list = dlib_detector(img_gray, 1)
    face_area_list = []
    for rect in rect_list:
        (h, w) = (rect.height(), rect.width())
        (h, w) = (int(h * enlarge), int(w * enlarge))
        top = rect.top() - h if rect.top() - h > 0 else 0
        bottom = rect.bottom() + h if rect.bottom() + h < imgArr.shape[0] else imgArr.shape[0]
        left = rect.left() - h if rect.left() - h > 0 else 0
        right = rect.right() + h if rect.right() + h < imgArr.shape[1] else imgArr.shape[1]
        face_area_list.append((imgArr[top:bottom, left:right], rect.area()))
    face_area_list = sorted(face_area_list, key=lambda x: x[1], reverse=True)
    return face_area_list


# In[ ]:


sample_dir="./tmp/CV_clf/ethnicity"
label = ['Australoid','Negroid','Caucasoid','Mongoloid']
targetFormat = ['.jpg']
for label_dir in [os.path.join(sample_dir,i) for i in label]:
    for root, dirs, files in os.walk(label_dir):
        face_root = root.replace("ethnicity","ethnicity/face")
        no_face_root = root.replace("ethnicity","ethnicity/no_face")
        if not os.path.exists(face_root): os.mkdir(face_root)
        if not os.path.exists(no_face_root): os.mkdir(no_face_root)
        for name in tqdm(files,desc=root):
            if not name.startswith(".") and os.path.splitext(name)[-1] in targetFormat:
                imgArr = np.array(Image.open(os.path.join(root,name)))
                save_name = os.path.splitext(name)[0]+"_face{}"+os.path.splitext(name)[-1]
                save_path = os.path.join(face_root,save_name)
                face_area_list = get_face_imgArr(imgArr)
                if len(face_area_list)>0:
                    for idx,(faceArr,area) in enumerate(face_area_list):
                        Image.fromarray(faceArr).save(save_path.format(idx))
                else:
                    Image.fromarray(imgArr).save(os.path.join(no_face_root,name))


# ## 数据流

# ### 公开数据集 | 花草数据

# In[ ]:


IMAGE_SHAPE = (96,96)
batch_size = 32
validation_ratio=0.1
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

ig_flower = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255)
label = [i for i in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,i))]
generic_params = dict(directory=str(data_root), classes=label, target_size=IMAGE_SHAPE, batch_size=batch_size)
# image_data
normal_flow_train = ig_flower.flow_from_directory(subset='training', **generic_params)
normal_flow_valid = ig_flower.flow_from_directory(subset='validation', **generic_params)

for image_batch, label_batch in normal_flow_train:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

# 重新加载一下iter
normal_flow_train.reset()
print(label)


# ### dlib效果

# In[ ]:


def get_face_list_from_pil(imgPIL, dlib_detector, enlarge=0.2, target_size=(224, 224), scale=1):
    imgArr = np.array(imgPIL)
    gray_img = np.array(imgPIL.convert("L"))
    res_list = dlib_detector(gray_img, scale)
    face_img_area_list = []
    for res in res_list:
        if isinstance(dlib_detector,dlib.fhog_object_detector):
            rect = res
        elif isinstance(dlib_detector,dlib.cnn_face_detection_model_v1):
            rect = res.rect
        else:
            assert False,"unkown type of dlib_detector: %s"%type(dlib_detector)
        (h, w) = (rect.height(), rect.width())
        (h, w) = (int(h * enlarge), int(w * enlarge))
        top = rect.top() - h if rect.top() - h > 0 else 0
        bottom = rect.bottom() + h if rect.bottom() + h < imgArr.shape[0] else imgArr.shape[0]
        left = rect.left() - h if rect.left() - h > 0 else 0
        right = rect.right() + h if rect.right() + h < imgArr.shape[1] else imgArr.shape[1]
        facePIL = Image.fromarray(imgArr[top:bottom, left:right]).resize(target_size)
        face_img_area_list.append((np.array(facePIL), rect.area()))
    face_img_list = [img for img,area in sorted(face_img_area_list, key=lambda x: x[1], reverse=True)]  # 多个人脸按人脸面积大到小排序
    return np.array(face_img_list)

def pltshow_images(img_list, limit=36, figsize=(10, 10), fig_title=""):
    total = len(img_list) if len(img_list) <= limit else limit
    c, r = int(total**0.5), int(total**0.5)+2
    fig, axes = plt.subplots(c, r, figsize=figsize)
    fig.suptitle(fig_title)
    for idx, img in enumerate(img_list):
        axe = axes.flatten()[idx]
        _ = axe.imshow(img)
    return fig, axes

from zac_pyutils import CVUtils
# url = "https://static.toiimg.com/photo/65798582.cms"
# url = "https://thumbor.apusapps.com/imageView2/material/7ad3fc32/201912/032204/52fc26d0b2e84b428832c98e5720a68d.jpg?mode=0&w=300&h=300"
# url = "https://www.google.com/search?biw=1733&bih=905&tbm=isch&sxsrf=ACYBGNS5lK_8OVUGuLzWRdDRPhTzU6vIag%3A1577182760047&sa=1&ei=KOYBXqa3AtKC-QaY-augBA&q=indian+woman&oq=indian+woman&gs_l=img.3..35i39.8774.9155..9384...0.0..0.307.1573.0j7j0j1......0....1..gws-wiz-img.......0i7i30j0i30j0.qsmWfAcFs1o&ved=0ahUKEwim66HRh87mAhVSQd4KHZj8CkQQ4dUDCAc&uact=5#imgrc=oj7-5O25VEvkKM:"
url = "https://thumbor.apusapps.com/imageView2/material/7ad3fc32/201912/271153/9728b50113d84517a6e85b0a68852d6b.jpg?mode=0&w=800&h=800"

from urllib.parse import urlparse, parse_qs, urlencode
parts = urlparse(url)

build_url = parts._replace(query=urlencode({"mode":0,"w":800,"h":800})).geturl()
url = build_url

imgPIL = CVUtils.Load.image_by_pil_from(url)
imgArr = np.array(imgPIL)
detectorCNN=dlib.cnn_face_detection_model_v1("./tmp/CV_clf/mmod_human_face_detector.dat")
detector=dlib.get_frontal_face_detector()
imgPIL
imgPIL.size
face_list = get_face_list_from_pil(imgPIL, dlib_detector = detectorCNN)
face_list_hog = get_face_list_from_pil(imgPIL, dlib_detector = detector, scale=2)
print(f"dlib (CNN) 总计检测到人脸: {len(face_list)} 个")
get_ipython().run_line_magic('timeit', 'get_face_list_from_pil(imgPIL, dlib_detector = detectorCNN)')
print(f"dlib (HOG) 总计检测到人脸: {len(face_list_hog)} 个")
get_ipython().run_line_magic('timeit', 'get_face_list_from_pil(imgPIL, dlib_detector = detector, scale=2)')

fig,axes = pltshow_images(face_list,fig_title="dlib (CNN)")
for axe in axes.flatten():
    axe.set_axis_off()
    
fig,axes = pltshow_images(face_list_hog, fig_title="dlib (HOG)")
for axe in axes.flatten():
    axe.set_axis_off()


# ### 人种数据

# 人脸检测效果

# In[ ]:


import dlib

detectorCNN=dlib.cnn_face_detection_model_v1("./tmp/CV_clf/mmod_human_face_detector.dat")
detector=dlib.get_frontal_face_detector()

def get_faceArr_from_PIL(imgPIL, dlib_detector=detector, enlarge=0.2):
    imgArr = np.array(imgPIL)
    gray_img = np.array(imgPIL.convert("L"))
    rect_list = dlib_detector(gray_img, 1)
    face_img_list = []
    for rect in rect_list:
        (h, w) = (rect.height(), rect.width())
        (h, w) = (int(h * enlarge), int(w * enlarge))
        top = rect.top() - h if rect.top() - h > 0 else 0
        bottom = rect.bottom() + h if rect.bottom() + h < imgArr.shape[0] else imgArr.shape[0]
        left = rect.left() - h if rect.left() - h > 0 else 0
        right = rect.right() + h if rect.right() + h < imgArr.shape[1] else imgArr.shape[1]
#         facePIL = Image.fromarray(imgArr[top:bottom, left:right])
        face_img_list.append(imgArr[top:bottom, left:right])
    return face_img_list


sample_dir = "./tmp/CV_clf/ethnicity/face/origin"
label = ['Indian','Negroid','Caucasoid','Mongoloid']

for root, dirs, files in os.walk(sample_dir):
    if os.path.split(root)[-1] in label:
        for name in tqdm(files,desc=root):
            if not name.startswith(".") and name.split(".")[-1] in ["png","jpg"]:
                path = os.path.join(root,name)
                imgPIL = Image.open(path)
#                 imgPIL
                path_new = path.replace("face/origin","face/dlib_frontal")
                for idx,i in enumerate(get_faceArr_from_PIL(imgPIL, dlib_detector=detector, enlarge=0.1)):
                    save_p = os.path.splitext(path_new)[0]+"_f%s"%idx+os.path.splitext(path_new)[1]
                    Image.fromarray(i).save(save_p)
#                     save_p
#                 break


# In[11]:


# IMAGE_SHAPE = (96,96)
IMAGE_SHAPE = (224,224)
# IMAGE_SHAPE = (299,299)
batch_size = 32
validation_ratio=0.2
sample_dir = "./tmp/CV_clf/ethnicity/face/dlib_frontal_0.25"
service=sample_dir.split("CV_clf/")[1].split("/")[0]
label = ['Caucasoid','Mongoloid','Indian','Negroid']
# label = ['Caucasoid','Mongoloid']
# label = ['Indian','Negroid']
# label = ['CauMon','IndNeg']

augment_params={"horizontal_flip":True, "rotation_range":90}
# augment_params={}

ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255,**augment_params)

generic_params = dict(directory=sample_dir, classes=label, target_size=IMAGE_SHAPE, batch_size=batch_size)
normal_flow_train = ig_normal.flow_from_directory(subset='training', **generic_params)
normal_flow_valid = ig_normal.flow_from_directory(subset='validation', **generic_params)

for image_batch, label_batch in normal_flow_train:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
print(label)
# 重新加载一下iter
normal_flow_train.reset()
print(">>> 训练集分布：")
print(np.vstack(np.unique([i.split("/")[-2] for i in normal_flow_train.filepaths],return_counts=True)).T)
print(">>> 测试集分布：")
print(np.vstack(np.unique([i.split("/")[-2] for i in normal_flow_valid.filepaths],return_counts=True)).T)


# ### NSFW数据

# In[62]:


# IMAGE_SHAPE = (96,96)
IMAGE_SHAPE = (224,224)
# IMAGE_SHAPE = (299,299)
batch_size = 32
validation_ratio=0.2
sample_dir = "./tmp/CV_clf/nsfw/samples"
service=sample_dir.split("CV_clf/")[1].split("/")[0]
label = ['NSFW','SFW']


augment_params={"horizontal_flip":True, "rotation_range":90}
# augment_params={}

ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255,**augment_params)

generic_params = dict(directory=sample_dir, classes=label, target_size=IMAGE_SHAPE, batch_size=batch_size)
normal_flow_train = ig_normal.flow_from_directory(subset='training', **generic_params)
normal_flow_valid = ig_normal.flow_from_directory(subset='validation', **generic_params)

for image_batch, label_batch in normal_flow_train:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
print(label)
# 重新加载一下iter
normal_flow_train.reset()
print(">>> 训练集分布：")
print(np.vstack(np.unique([i.split("/")[-2] for i in normal_flow_train.filepaths],return_counts=True)).T)
print(">>> 测试集分布：")
print(np.vstack(np.unique([i.split("/")[-2] for i in normal_flow_valid.filepaths],return_counts=True)).T)


# ## 临时 | 用SVM试试
# 多分类SVM还是不好弄
# 
# 聚类：KMeans的聚类中心做向量相似召回；GMM直接获得打分
# 

# ### sklearn.SVC

# In[ ]:


clf = SVC()
# clf.fit()


# ## 模型 

# ### Subclass + keras.applications

# In[ ]:


class EthnicityM_KApp(tf.keras.Model):
    def __init__(self,num_classes:int, scope:str="EthnicityM", **kwargs):
        super().__init__(name="EthnicityM", **kwargs)
        # 参考自： https://www.tensorflow.org/tutorials/images/transfer_learning
        # MobileNetV2 获得的是 (None, 3, 3, 1280) 的特征block
        self.feature_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE+(3,), include_top=False,weights="imagenet")
        self.feature_model.trainable=False
        # 平均池化 得到 (None, 1280) 的特征
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # 全连接
        self.prediction_layer = tf.keras.layers.Dense(num_classes)
        
    def call(self, inp, training=False):
        x = self.feature_model(inp)
        x = self.global_average_layer(x)
        x = self.prediction_layer(x)
        return x

model=EthnicityM_KApp(normal_flow_train.num_classes)
model.build((None,96,96,3))
model.summary()

checkpoint_dir = "./tmp/CV_clf/{}/ckpt_Subclass_KApp".format(service)


# ### Subclass + Hub

# In[57]:


class EthnicityM_Hub(tf.keras.Model):
    def __init__(self,input_shape_, num_classes:int, fine_tuning:bool=False, scope:str=None, rate:float=0.1, **kwargs):
        if scope is None:
            scope=self.__class__.__name__
        super().__init__(name=scope, **kwargs)
        self.input_shape_ = input_shape_
        tf.keras.backend.set_learning_phase(True)
        feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
        self.feature_model = hub.KerasLayer(feature_extractor_url, input_shape=self.input_shape_,trainable=fine_tuning)
        self.feature_model.trainable=fine_tuning
        
        # 全连接
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.prediction_layer = tf.keras.layers.Dense(num_classes, 
                                                      activation='softmax', 
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.FC=self.prediction_layer
        
    def call(self, inp, training=False):
        x = self.feature_model(inp)
        x = self.dropout1(x, training=training)
        x = self.prediction_layer(x)
        return x

    def summary(self):
        x = tf.keras.Input(shape=self.input_shape_)
        tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


model=EthnicityM_Hub(input_shape_=IMAGE_SHAPE+(3,), num_classes=normal_flow_train.num_classes, fine_tuning=True)
model.build((None,)+IMAGE_SHAPE+(3,))
model.summary()
# model.model().summary()

# checkpoint_dir = "./tmp/CV_clf/ethnicity/ckpt_Subclass_Hub"
# checkpoint_dir = "./tmp/CV_clf/nsfw/ckpt_Subclass_Hub"
checkpoint_dir = "./tmp/CV_clf/{}/ckpt_Subclass_Hub".format(service)


# ### Sequential + Hub

# In[52]:


do_fine_tunning=True
def get_sequential_model(fine_tunning=False):
#     feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4"
#     feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4" #@param {type:"string"}
    feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
    feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
    feat_layer.trainable = fine_tunning  # feature_vector的生成就不用训练了
    return tf.keras.Sequential([
      feat_layer,
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(normal_flow_train.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
model = get_sequential_model(fine_tunning=do_fine_tunning)
model.summary()

checkpoint_dir = f"./tmp/CV_clf/{service}/ckpt_Sequential_Hub{'_fintunnig' if do_fine_tunning else ''}"
checkpoint_dir


# 先训练FC层（freeze其他），然后再（加载出FC训练过的权重）一起做fine-tuning
# 没什么显著效果
# 
# **【注】当初warmup FC的时候学习率没有控制的很好，应该用很小的学习率**

# **模型的Dense层重置权重后重新训练**
# 
# 此尝试效果很差， 训练集acc loss提升很好，但是验证集提升很慢

# In[ ]:


ckpt_toload=tf.train.latest_checkpoint("./tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Sequential_Hub_fintunnig_loss1.2564_acc0.7099")
print(f"加载ckpt {ckpt_toload}")
model.load_weights(ckpt_toload)

l1 = model.layers[1]
l1.set_weights(weights=[l1.kernel_initializer(l1.weights[0].shape), l1.bias_initializer(l1.weights[1].shape)])
model.layers[1].weights
model.summary()


# ### bcnn | Subclass + KerasApp

# In[82]:


do_fine_tunning=True
class EthnicityM_BCNN_App(tf.keras.Model):
    def __init__(self, num_classes:int, fine_tuning:bool=False, scope:str=None, rate:float=0.4, **kwargs):
        if scope is None:
            scope = self.__class__.__name__
        super().__init__(scope, **kwargs)
        # 参考自： https://github.com/abhaydoke09/Bilinear-CNN-TensorFlow/blob/master/core/bcnn_DD_woft.py
        vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        self.featM = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output)
        self.featM.trainable=fine_tuning
        
        # 全连接
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.prediction_layer = tf.keras.layers.Dense(num_classes, 
                                                      activation='softmax', 
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.FC = self.prediction_layer
    
    @tf.function 
    def bilinear(self,inp):
        # bilinear pool
        phi_I = tf.einsum('ijkm,ijkn->imn',inp,inp)  # 外积
        phi_I = tf.reshape(phi_I,[-1,512*512])
        phi_I = tf.divide(phi_I, inp.shape[1]*inp.shape[2])  # 归一化
        y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))  # sing、开方计算
        feat = tf.nn.l2_normalize(y_ssqrt, axis=1)  # L2正则化
        return feat

    @tf.function
    def bilinear_old(self,conv5_3):
        # ERROR 这个方法算出的结果 shape不对
        conv5_3 = tf.transpose(conv5_3, perm=[0,3,1,2])
        conv5_3 = tf.reshape(conv5_3,[-1,512,conv5_3.shape[1]*conv5_3.shape[2]])
        conv5_3_T = tf.transpose(conv5_3, perm=[0,2,1])
        phi_I = tf.matmul(conv5_3, conv5_3_T)
        
        phi_I = tf.reshape(phi_I,[-1,512*512])
        phi_I = tf.divide(phi_I, conv5_3.shape[1]*conv5_3.shape[2])  # 归一化
        y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))  # sing、开方计算
        feat = tf.nn.l2_normalize(y_ssqrt, axis=1)  # L2正则化
        return feat
        
    def call(self, inp, training=False):
        x = self.featM(inp)
        x = self.bilinear(x)
        x = self.dropout1(x, training=training)
        x = self.prediction_layer(x)
        return x
    
    def build(self, input_shape, *args, **kwargs):
        self.inp_shape=input_shape[1:] # idx=0元素是batch_size，不取
        super().build(input_shape,*args,**kwargs)
        
    def summary(self):
        if type(self.bilinear).__name__ == "method":
            x = tf.keras.Input(shape=self.inp_shape)
            tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()
        else:
            print("WARN: bilinear操作不是普通的pyfunc，不支持直接接入tf.keras.Model")
            super().summary()


model=EthnicityM_BCNN_App(num_classes=normal_flow_train.num_classes, fine_tuning=do_fine_tunning)
model.build((None,)+IMAGE_SHAPE+(3,))
model.summary()


checkpoint_dir = "./tmp/CV_clf/ethnicity/ckpt_{}".format(type(model).__name__)
checkpoint_dir = "./tmp/CV_clf/nsfw/ckpt_{}".format(type(model).__name__)
checkpoint_dir = "./tmp/CV_clf/{}/ckpt_{}".format(service,type(model).__name__)

checkpoint_dir


# In[11]:


use_ckpt = tf.train.latest_checkpoint("/home/zhoutong/notebook_collection/tmp/CV_clf/{}/archived_ckpts/ckpt_EthnicityM_BCNN_App_acc0.545".format(service))
print("load ckpt: ",use_ckpt)
model.load_weights(use_ckpt)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])
model.evaluate_generator(normal_flow_valid)


# ### inceptionv3 | SubclassedM + KerasApp

# In[18]:


class EthnicityM_inceptionV3_App(tf.keras.Model):
    def __init__(self, num_classes:int, fine_tuning:bool=False, scope:str=None, rate:float=0.1, **kwargs):
        if scope is None:
            scope=self.__class__.__name__
        super().__init__(name=scope, **kwargs)
        tf.keras.backend.set_learning_phase=True
        # featM
        xcep = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',include_top=False)
        self.featM = tf.keras.Model(inputs=xcep.input, outputs=xcep.layers[-1].output)
        self.featM.trainable=fine_tuning
        # avg pool
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # 全连接
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.prediction_layer = tf.keras.layers.Dense(num_classes, 
                                                      activation='softmax', 
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.FC=self.prediction_layer
        
    def call(self, inp, training=False):
        x = self.featM(inp)
        x = self.avg_pool(x)
        x = self.dropout1(x, training=training)
        x = self.prediction_layer(x)
        return x
    
    def build(self, input_shape, *args, **kwargs):
        self.inp_shape=input_shape[1:] # idx=0元素是batch_size，不取
        super().build(input_shape,*args,**kwargs)
        
    def summary(self):
        x = tf.keras.Input(shape=self.inp_shape)
        tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


model=EthnicityM_inceptionV3_App(num_classes=normal_flow_train.num_classes, fine_tuning=True)
model.build((None,)+IMAGE_SHAPE+(3,))
model.summary()
# model.model().summary()

checkpoint_dir = "./tmp/CV_clf/ethnicity/ckpt_Subclass_KerasApp"
checkpoint_dir = "./tmp/CV_clf/{}/ckpt_Subclass_KerasApp".format(service)


# ## 训练

# ### 自己写循环

# #### warm up FC
# - 配置了EarlyStop

# In[83]:


model.featM.trainable=False
model.summary()
checkpoint_path = os.path.join(checkpoint_dir,"warm_fc","ckpt_e{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}")
ckpt_cb=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                           monitor="val_acc",
                                           save_weights_only=True, 
                                           verbose=1, save_best_only=True)
es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)

normal_flow_train.reset()
normal_flow_train.shuffle=True
normal_flow_valid.reset()
normal_flow_valid.shuffle=True
print(f"use batch_size: {normal_flow_train.batch_size}")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=['acc'])
history = model.fit_generator(normal_flow_train, epochs=100,
                              steps_per_epoch = 1.5*normal_flow_train.samples // normal_flow_train.batch_size,
                              validation_data = normal_flow_valid,
                              verbose=1,
                              callbacks=[ckpt_cb,es_cb])

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig,axes_list = plt.subplots(2,1, figsize=(8,8))
axe1=axes_list.flatten()[0]
axe2=axes_list.flatten()[1]

_ = axe1.plot(acc, label='Training Accuracy')
_ = axe1.plot(val_acc, label='Validation Accuracy')
_ = axe1.legend(loc='lower right')
_ = axe1.set_ylabel('Accuracy')
_ = axe1.set_ylim([min(plt.ylim()),1])
_ = axe1.set_title('Training and Validation Accuracy')

_ = axe2.plot(loss, label='Training Loss')
_ = axe2.plot(val_loss, label='Validation Loss')
_ = axe2.legend(loc='upper right')
_ = axe2.set_ylabel('Cross Entropy')
_ = axe2.set_ylim([0,2.0])
_ = axe2.set_title('Training and Validation Loss')
_ = axe2.set_xlabel('epoch')

tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))


# In[85]:


print(">>> 加载权重前")
conv=[i for i in model.featM.layers if "Conv2D" in type(i).__name__][0]
try:
    bn=[i for i in model.featM.layers if type(i).__name__ =="BatchNormalization"][0]
except IndexError as e:
    bn=model.featM.layers[-3]
fc = model.FC
[(i.name, np.mean(i.numpy()),np.std(i.numpy())) for i in conv.variables+bn.variables+fc.variables]

ckpt_dir = os.path.dirname(checkpoint_path)
# ckpt_dir = "./tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Subclass_Hub_bcnn_acc0.4578_onlyFC"
tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(tf.train.latest_checkpoint(ckpt_dir))

print(">>> 加载权重后")
[(i.name, np.mean(i.numpy()),np.std(i.numpy())) for i in conv.variables+bn.variables+fc.variables]


# In[86]:


model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])
model.evaluate_generator(normal_flow_valid)


# In[87]:


normal_flow_valid.reset()
normal_flow_valid.shuffle = False
tf.reduce_mean(tf.metrics.categorical_accuracy(model.predict_generator(normal_flow_valid), tf.one_hot(normal_flow_valid.labels,depth=len(label))))


# #### tbd ckpt ..

# tensorboard
# 
# ckpt
# - ckpt还需要知道opt和model? （推测应该是不指定无法resotre）
# 
# 看看官方手册
# - [完整ckpt & ckpt-manager 的示例](https://www.tensorflow.org/guide/checkpoint)
# - [checkpoint类](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)

# In[92]:


# tensorboard写入第一组图片
# tbd_dir = "./tmp/CV_clf/ethnicity/tensorboard"
# tbd_dir = "./tmp/CV_clf/nsfw/tensorboard"
tbd_dir = "./tmp/CV_clf/{}/tensorboard".format(service)
if os.path.exists(tbd_dir):
    import shutil
    shutil.rmtree(tbd_dir)
    print("历史tbd信息已删除")
    
summary_writer = tf.summary.create_file_writer(tbd_dir)
with summary_writer.as_default():
    _=tf.summary.image("Trainning Data", normal_flow_train[0][0], max_outputs=4, step=0)

# 为了写入计算图
# @tf.function
# def traceme(x):
#     return model(x)
# tf.summary.trace_on(graph=True, profiler=True)
# traceme(tf.zeros((1,)+normal_flow_train.image_shape))
# with summary_writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir="./tmp/CV_clf/ethnicity/tensorboard")


print("ckpt不能存模型内部还有Model类变量的模型，直接改用model.save_weights")
# ckpt = tf.train.Checkpoint(step=tf.Variable(1),model=model)
checkpoint_path = os.path.join(checkpoint_dir,"ckpt_e{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}")
print("ckpt_manager 不能在保存时指定prefix 不用它了")
# ckpt = tf.train.Checkpoint(opt=opt,model=model)
# status = ckpt.restore(tf.train.latest_checkpoint(checkpoint_directory))
# ckpt.save(file_prefix=checkpoint_prefix)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
checkpoint_path


# #### fine-tuning

# In[93]:


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
ce_loss_fn = tf.keras.losses.categorical_crossentropy
# mean_calc = tf.keras.metrics.Mean()  # 这个Mean是累积的，也就是说到了第100batch它计算的实际上是0~100batch的均值 | 所以用它需要清除一下state
acc_fn = tf.keras.metrics.categorical_accuracy

# @tf.function  #| 加装饰器会报错，不加的话，这种函数式封装会让训练慢一倍
def train(model_inp, optimizer_inp, step, step_per_epoch):
    # 第一个epoch记录100个step，后续每个epoch只记录10次
    assert step_per_epoch >=100
    with tf.GradientTape() as tape:
        pred_batch = model_inp(image_batch, training=True)
        loss_batch = ce_loss_fn(label_batch,pred_batch)
        acc_batch = acc_fn(label_batch,pred_batch)
    gradients = tape.gradient(loss_batch, model_inp.trainable_variables)
    _ = optimizer_inp.apply_gradients(zip(gradients, model_inp.trainable_variables))
    b_loss, b_acc = tf.math.reduce_mean(loss_batch),tf.math.reduce_mean(acc_batch)
    if (e==0 and step % (step_per_epoch//100)==0) or (e>0 and step % (step_per_epoch//10) == 0):
        _ = tf.summary.scalar('train_loss', b_loss, step=step+e*step_per_epoch)
        _ = tf.summary.scalar('train_acc', b_acc, step=step+e*step_per_epoch)
    for i in model.weights:
        if "moving" in i.name:
            _ = tf.summary.histogram(i.name,i, step=step+e*step_per_epoch)
    for i in model.weights:
        _ = tf.summary.histogram(i.name,i, step=step+e*step_per_epoch)
#     for i in gradients:  | 保存梯度会报错？
#         _ = tf.summary.histogram(i.name,i, step=step+e*step_per_epoch, description=f"grad of {i.name}")
    return loss_batch, acc_batch


# In[94]:


model.featM.trainable=True
model.summary()
normal_flow_train.reset()
normal_flow_train.shuffle=True
step_per_epoch = 2* normal_flow_train.samples // normal_flow_train.batch_size
best_valid_acc = 0.0
best_valid_loss=1e10
history = []
max_ckpt_keep = 20
total_save_ckpt_fp = []

print(">>> fine-tuning")
for e in range(200):
    e_loss, e_acc = [], []
    for step, (image_batch, label_batch) in tqdm(enumerate(normal_flow_train), desc=f"Epoch: {e}", total=step_per_epoch, leave=False):
        if step>= step_per_epoch:
            break
        with summary_writer.as_default():
            loss_batch, acc_batch = train(model, optimizer, step, step_per_epoch)
            e_loss.append(np.mean(loss_batch))
            e_acc.append(np.mean(acc_batch))
           
    normal_flow_valid.reset()
    normal_flow_valid.shuffle=False
    val_pred = model.predict_generator(normal_flow_valid)
    print(">>> 验证集predict结果如下：\n",val_pred)
    val_label = tf.one_hot(normal_flow_valid.labels,depth=normal_flow_valid.num_classes)
    val_loss = np.mean(ce_loss_fn(val_label, val_pred).numpy())
    val_acc = np.mean(acc_fn(val_label, val_pred).numpy())
    with summary_writer.as_default(): 
        _ = tf.summary.scalar('val_loss', val_loss, step=step+e*step_per_epoch)
        _ = tf.summary.scalar('val_acc', val_acc, step=step+e*step_per_epoch)
    print(f'[e]:{e} [step]:{step} [e-loss]:{np.mean(e_loss):.4f} [e-acc]:{np.mean(e_acc):.4f} [val_loss]:{val_loss:.4f} [val_acc]:{val_acc:.4f}')
    
    saved=False
    save_path = checkpoint_path.format(epoch=e,val_loss=val_loss,val_acc=val_acc)
    if val_acc > best_valid_acc:
        print(f"acc improved [from]:{best_valid_acc:.4f} [to]:{val_acc:.4f}.")
        best_valid_acc = val_acc
        if not saved:
            model.save_weights(save_path)
            saved=True
    if val_loss < best_valid_loss:
        print(f"loss improved [from]:{best_valid_loss:.4f} [to]:{val_loss:.4f}.")
        best_valid_loss = val_loss
        if not saved:
            model.save_weights(save_path)
            saved=True
    if saved:
        print(f"[ckpt-path]: {save_path}")
        total_save_ckpt_fp.append(save_path)
        if len(total_save_ckpt_fp) >= max_ckpt_keep:
            toDel_fp = total_save_ckpt_fp.pop(0)
            status,output=subprocess.getstatusoutput(f"rm {toDel_fp}*")


    history.append({"val_acc":val_acc, "val_loss":val_loss, "FC_kernel":model.FC.kernel.numpy()})


# In[95]:


print(">>> 加载权重前")
conv=[i for i in model.featM.layers if "Conv2D" in type(i).__name__][0]
try:
    bn=[i for i in model.featM.layers if type(i).__name__ =="BatchNormalization"][0]
except IndexError as e:
    bn=model.featM.layers[-3]
fc = model.FC
[(i.name, np.mean(i.numpy()),np.std(i.numpy())) for i in conv.variables+bn.variables+fc.variables]
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])
model.evaluate_generator(normal_flow_valid)

ckpt_dir = os.path.dirname(checkpoint_path)
# ckpt_dir = "./tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Subclass_Hub_bcnn_acc0.4578_onlyFC"
# ckpt_dir = "/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Subclass_KerasApp_acc0.71"
use_ckpt = tf.train.latest_checkpoint(ckpt_dir)
print("load ckpt: ",use_ckpt)
model.load_weights(use_ckpt)

print(">>> 加载权重后")
[(i.name, np.mean(i.numpy()),np.std(i.numpy())) for i in conv.variables+bn.variables+fc.variables]
model.evaluate_generator(normal_flow_valid)


# #### 检查第一个batch的梯度

# In[ ]:


print(f">>> model可训练的变量有{len(model.trainable_variables)}个, FC层的kernel和bias如下:")
print(">>> kernel\n",model.prediction_layer.kernel)
k_ori = model.prediction_layer.kernel.numpy().flatten()
b_ori = model.prediction_layer.bias.numpy().flatten()
print(">>> bias\n",model.prediction_layer.bias)
normal_flow_train.reset()
normal_flow_train.shuffle=False
image_batch = normal_flow_train.next()
print(">>> 展示第一个batch的首图(注意可能有augment需要到加载数据时关掉):")
_ = plt.imshow(image_batch[0][0])
plt.show()
with tf.GradientTape(persistent=True) as tape:
    pred_batch = model(image_batch)
    loss_batch = ce_loss_fn(label_batch,pred_batch)
    acc_batch = acc_fn(label_batch,pred_batch)
gradients = tape.gradient(loss_batch, model.trainable_variables)
print(f">>> 第一个batch的梯度shape:{gradients[0].shape}:")
_ = [print(f"'{model.trainable_variables[idx].name}' -- {i}") for idx,i in enumerate(gradients)]
_ = optimizer.apply_gradients(zip(gradients, model.trainable_variables))

print(">>> 更新梯度后的kernel\n",model.prediction_layer.kernel)
k = model.prediction_layer.kernel.numpy().flatten()
b = model.prediction_layer.bias.numpy().flatten()
print(f"mean:{np.mean(k)}, std:{np.std(k)}")
print(f"max:{np.max(k)}, min:{np.min(k)}")
print(f"p0.99:{np.percentile(k,99)}, p0.95:{np.percentile(k,95)}, p0.90:{np.percentile(k,90)} ")
print(f"p0.75:{np.percentile(k,75)}, p0.35:{np.percentile(k,35)}")
print(f"p0.15:{np.percentile(k,15)}, p0.05:{np.percentile(k,5)}, p0.01:{np.percentile(k,1)}")

def plot(k,k_ori,range_=None,name=""):
    if range_ is None:
        range_ = (np.percentile(k,10), np.percentile(k,90))
    fig,axs_up = plt.subplots(2,2,figsize=(10,10))
    _ = [ax.set_axis_off() for ax in axs_up.flatten()]
    _=fig.suptitle(f"fc's {name} distribution")
    _ = axs_up[0,0].set_title("before update")
    _ = axs_up[0,0].set_axis_on()
    _ = axs_up[0,0].hist(k_ori,bins=2000,range=range_,histtype='step',color='red')
    _ = axs_up[0,1].set_title("after update")
    _ = axs_up[0,1].set_axis_on()
    _ = axs_up[0,1].hist(k,bins=2000,range=range_,histtype='step',color='blue')
    ax_down = fig.add_subplot(2,1,2)
    _=ax_down.set_title("merge compare")
    _=ax_down.hist(k_ori,bins=2000,range=range_,histtype='step',color='red',density=True)
    _=ax_down.hist(k,bins=2000,range=range_,histtype='step',color='blue', density=True)
    return fig
_ = plot(k,k_ori,name="kernel")
_ = plot(b,b_ori,name="bias")
del tape


# ### 使用Keras的 fit
# 
# CPU & mobilenet_v2_075_96 & batch_size=1 592张验证集的图耗时390s
# 
# GPU & mobilenet_v2_075_96 & batch_size=1 592张验证集的图耗时479s
# 
# GPU & inception_resnet_v2 & batch_size=1 592张验证集的图耗时 ? s (训练8010张图在两小时以内）
# 
# GPU & inception_v3 & batch_size=16 667个训练batch耗时142s

# In[61]:


####################
# 初始化各种callback 
###################
checkpoint_path = os.path.join(checkpoint_dir,"ckpt_e{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}")
# official callback
ckpt_cb=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                           monitor="val_acc",
                                           save_weights_only=True, 
                                           verbose=1, save_best_only=True)
# LambdaCallback
best_acc=0.0
best_loss=sys.float_info.max
def saveCkpt(epoch,logs):
    val_acc = logs['val_acc']
    val_loss = logs['val_loss']
    save_path = checkpoint_path.format(epoch=epoch,val_loss=val_loss,val_acc=val_acc)
    if val_acc > best_acc:
        print(f"val_acc improved from {best_acc:.4f} to {val_acc:.4f}, ckpt saved to {save_path}")
        best_acc = val_acc
        model.save_weights(save_path)
    else:
        if val_loss < best_loss:
            print(f"val_loss improved from {best_loss:.4f} to {val_loss:.4f}, ckpt saved to {save_path}")
            best_loss=val_loss
            model.save_weights(save_path)
ckpt_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=saveCkpt)

# CustomCallback
class SaveCkpt(tf.keras.callbacks.Callback):
    def __init__(self, filepath, max_=10):
        super(SaveCkpt, self).__init__()
        self.best_acc=0.0
        self.best_loss=1e10
        self.max = max_
        self.checkpoint_path = checkpoint_path
        self.saved_path_list=[]
        
    def on_epoch_end(self, epoch,logs):
        val_acc = logs['val_acc']
        val_loss = logs['val_loss']
        save_path = self.checkpoint_path.format(epoch=epoch,val_loss=val_loss,val_acc=val_acc)
        not_save = True
        if val_acc > self.best_acc:
            # best_acc只要有更优的就更新，存不存另说
            if not_save:
                print(f"\n  [val_acc]: {self.best_acc:.4f} to {val_acc:.4f} [val_loss]: {self.best_loss:.4f} to {val_loss:.4f}\n  ckpt saved to {save_path}")
                self.model.save_weights(save_path)
                not_save = False
            self.best_acc = val_acc

        if val_loss < self.best_loss:
            # best_loss只要有更优的就更新，存不存另说
            if not_save:
                if self.best_loss>=1e10:
                    print(f"\n  [val_loss]: inf to {val_loss:.4f} [val_acc]: {self.best_acc:.4f} to {val_acc:.4f}\n  ckpt saved to {save_path}")
                else:
                    print(f"\n  [val_loss]: {self.best_loss:.4f} to {val_loss:.4f} [val_acc]: {self.best_acc:.4f} to {val_acc:.4f} \n  ckpt saved to {save_path}")
                self.model.save_weights(save_path)
                not_save = False
            self.best_loss=val_loss
        
        if not_save:
            print(f"\n  NOT Improved from [val_acc]: {self.best_acc:.4f} to {val_acc:.4f} [val_loss]: {self.best_loss:.4f} to {val_loss:.4f}")
        else:
            self.saved_path_list.append(save_path)
            if len(self.saved_path_list) >= self.max:
                todel=self.saved_path_list.pop(0)
                [os.remove(fp) for fp in os.listdir(os.path.dirname(todel)) if fp.startswith(os.path.split(todel)[-1])]  
                
ckpt_cb = SaveCkpt(filepath=checkpoint_path)
if os.path.exists(checkpoint_dir):
    import shutil
    shutil.rmtree(checkpoint_dir)
    print(f"历史ckpt信息已删除: {checkpoint_dir}")
else:
    print(f"ckpt at: {checkpoint_dir}")

# 使用Adam就用不上这个控制学习率的cb了
def decay(epoch):
    if  epoch <= 4:
        return 0.045
    elif 4 < epoch and epoch <= 10:
        return 1e-3
    else:
        return 1e-5
lr_cb = tf.keras.callbacks.LearningRateScheduler(decay)

# tbd_path = "./tmp/CV_clf/ethnicity/tensorboard/"
tbd_path = "./tmp/CV_clf/nsfw/tensorboard/"
if os.path.exists(tbd_path):
    import shutil
    shutil.rmtree(tbd_path)
    print(f"历史tbd信息已删除: {tbd_path}")
else:
    print(f"tbd at: {tbd_path}")
tbd_cb=tf.keras.callbacks.TensorBoard(log_dir=tbd_path,
                                      histogram_freq=1,write_graph=False,update_freq='batch',
                                      profile_batch=1)


# In[62]:


### # 模型训练 
##########
model
normal_flow_train.reset()
normal_flow_train.shuffle=True
normal_flow_valid.reset()
normal_flow_valid.shuffle=True
print(f"use batch_size: {normal_flow_train.batch_size}")
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])
history = model.fit_generator(normal_flow_train, epochs=180,
                              steps_per_epoch = 1.5*normal_flow_train.samples // normal_flow_train.batch_size,
                              validation_data = normal_flow_valid,
#                               validation_steps = normal_flow_valid.samples,
                              verbose=1,
                              callbacks=[ckpt_cb,tbd_cb])


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig,axes_list = plt.subplots(2,1, figsize=(8,8))
axe1=axes_list.flatten()[0]
axe2=axes_list.flatten()[1]

_ = axe1.plot(acc, label='Training Accuracy')
_ = axe1.plot(val_acc, label='Validation Accuracy')
_ = axe1.legend(loc='lower right')
_ = axe1.set_ylabel('Accuracy')
_ = axe1.set_ylim([min(plt.ylim()),1])
_ = axe1.set_title('Training and Validation Accuracy')


_ = axe2.plot(loss, label='Training Loss')
_ = axe2.plot(val_loss, label='Validation Loss')
_ = axe2.legend(loc='upper right')
_ = axe2.set_ylabel('Cross Entropy')
_ = axe2.set_ylim([0,2.0])
_ = axe2.set_title('Training and Validation Loss')
_ = axe2.set_xlabel('epoch')


# ## 模型 保存

# ### h5 moel.save
# - 不支持SubclassedModel

# In[68]:


# 要加载到最新（最优）的权重再导出存储
use_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
print("will load ckpt: ",use_ckpt)
h5_path = f"./tmp/CV_clf/nsfw/saved_models/{type(model).__name__}__loss{ckpt_cb.best_loss:.4f}_acc{ckpt_cb.best_acc:.4f}.h5"
print("will save at: ",h5_path)


# In[69]:


_ = model.load_weights(use_ckpt)
model.save(h5_path)


# ### pb keras.models.save_model

# In[102]:


use_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
print("will load ckpt: ",use_ckpt)
try:
    pb_path = f"./tmp/CV_clf/{service}/saved_models/{model.name}__loss{ckpt_cb.best_loss:.4f}_acc{ckpt_cb.best_acc:.4f}_pb"
except AttributeError as e:
    acc,loss=sorted([(kv['val_acc'],kv['val_loss']) for kv in history],key=lambda x:x[0], reverse=True)[0]
    pb_path = f"./tmp/CV_clf/{service}/saved_models/{model.name}__loss{loss:.4f}_acc{acc:.4f}_pb"
print("will save pb: ", pb_path)


# In[103]:


_ = model.load_weights(use_ckpt)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])
loss,acc = model.evaluate_generator(normal_flow_valid)
tf.keras.models.save_model(model, pb_path, save_format="tf")


# ## 模型 加载

# ### load ckpt

# In[15]:


# M = get_sequential_model()
M = EthnicityM_Hub(input_shape_=IMAGE_SHAPE+(3,), num_classes=normal_flow_train.num_classes)
# M = EthnicityM_inceptionV3_App(num_classes=normal_flow_train.num_classes)
M


# In[18]:


# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Sequential_Hub_inceptionv3_acc0.8746"
# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Sequential_Hub_fintunnig_loss1.2564_acc0.7099"
# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/ckpt_Sequential_Hub_fintunnig/"
# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/ckpt_Subclass_Hub"
# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/archived_ckpts/ckpt_Subclass_KerasApp_acc0.71"
checkpoint_dir_to_load = checkpoint_dir
checkpoint_dir_to_load = tf.train.latest_checkpoint(checkpoint_dir_to_load)
# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/ckpt_Sequential_Hub_fintunnig/ckpt_e88_loss1.5039_acc0.7279"
# checkpoint_dir_to_load="./tmp/CV_clf/ethnicity/ckpt_Sequential_Hub_fintunnig/ckpt_e58_loss1.0909_acc0.7200"


print(f"loading from ckpt: {checkpoint_dir_to_load}")
_ = M.load_weights(checkpoint_dir_to_load)
M.build((None,)+IMAGE_SHAPE+(3,))
M.count_params()
M.summary()

# 重置DENSE层的权重
# l1 = M.layers[1]
# l1.set_weights(weights=[l1.kernel_initializer(l1.weights[0].shape), l1.bias_initializer(l1.weights[1].shape)])
# l1.weights


# ### load h5

# In[ ]:


h5_path
M = tf.keras.models.load_model(h5_path,custom_objects={"KerasLayer":hub.KerasLayer})
M.count_params()
M

M.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])
loss,acc = M.evaluate_generator(normal_flow_valid)
print(loss,acc)


# ### load pb

# In[60]:


# pb_path = "./tmp/CV_clf/ethnicity/by_script_Mongoloid_Rest/saved_models/sequential_inceptionv3__loss0.5254_acc0.7969_pb"
# pb_path = "./tmp/CV_clf/ethnicity/by_script_subclassed_Indian_Caucasoid/saved_models/subclassed_inceptionv3__loss0.6389_acc0.7115_pb"
# pb_path = "./tmp/CV_clf/ethnicity/by_script_subclassed_Indian_Negroid/saved_models/subclassed_inceptionv3__loss0.4566_acc0.8562_pb"
# pb_path = "/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/by_script_Cau_Mon/saved_models/subclassed_inceptionv3_CauMon_vs_IndNeg__loss0.5010_acc0.8175_pb" # 24.8s
pb_path = "/home/zhoutong/notebook_collection/tmp/CV_clf/nsfw/saved_models/ethnicity_m_bcnn__app__loss0.2739_acc0.9418_pb"
print("load pb with: \n{}".format(pb_path))


# In[64]:


#M = tf.keras.models.load_model(pb_path,custom_objects={"KerasLayer":hub.KerasLayer})
M = tf.keras.models.load_model(pb_path)
M.count_params()

M.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=[tf.keras.metrics.categorical_accuracy])
loss,acc = M.evaluate_generator(normal_flow_valid)
print(loss,acc)


# ### load ensemble

# 加载pb

# In[6]:


archived_pb_dir = "/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/saved_models"
M_fps = ["Cau_vs_Mon_subclassed_inceptionv3__loss0.4173_acc0.8937_pb",
         "Ind_vs_Neg_subclassed_inceptionv3__loss0.4566_acc0.8562_pb",
         "CauMon_vs_IndNeg_subclassed_inceptionv3__loss0.4735_acc0.8118_pb"]
M1,M2,M3 = [tf.keras.models.load_model(os.path.join(archived_pb_dir,i)) for i in M_fps]
M1
M2
M3


# 重新搭建ensemble

# In[50]:


class EthnicityM_ensemble_tfn(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        base_dir="/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/saved_models/{}"
#         self.M1 = tf.keras.models.load_model(base_dir.format("CauMon_vs_IndNeg_subclassed_inceptionv3__loss0.4735_acc0.8118_pb"))
#         self.M2_0 = tf.keras.models.load_model(base_dir.format("Cau_vs_Mon_subclassed_inceptionv3__loss0.4173_acc0.8937_pb"))
#         self.M2_1 = tf.keras.models.load_model(base_dir.format("Ind_vs_Neg_subclassed_inceptionv3__loss0.4566_acc0.8562_pb"))
        self.label=np.array(['Caucasoid','Mongoloid','Indian','Negroid'])
        self.M1 = M1
        self.M2_0 = M2
        self.M2_1 = M3

    @tf.function()
    def call(self,inputs,training=False):
        p1 = tf.argmax(self.M1(inputs), axis=1)
        if p1==0:
            x = tf.pad(self.M2_0(inputs), [[0,0],[0,2]])
        else:
            x = tf.pad(self.M2_1(inputs), [[0,0],[2,0]])
        return x

M = EthnicityM_ensemble_tfn() # 2020-01-07 19:18:08 初始化此模型要 74s
M.build((None,)+(IMAGE_SHAPE)+(3,))
M.summary()


# In[13]:


class EthnicityM_ensemble_eager(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        base_dir="/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/saved_models/{}"
#         self.M1 = tf.keras.models.load_model(base_dir.format("CauMon_vs_IndNeg_subclassed_inceptionv3__loss0.4735_acc0.8118_pb"))
#         self.M2_0 = tf.keras.models.load_model(base_dir.format("Cau_vs_Mon_subclassed_inceptionv3__loss0.4173_acc0.8937_pb"))
#         self.M2_1 = tf.keras.models.load_model(base_dir.format("Ind_vs_Neg_subclassed_inceptionv3__loss0.4566_acc0.8562_pb"))
        self.label=np.array(['Caucasoid','Mongoloid','Indian','Negroid'])
        self.M1 = M1
        self.M2_0 = M2
        self.M2_1 = M3
        

    def call(self,inputs,training=False):
        x = self.M1(inputs).numpy()
        if x.argmax()==0:
            print(">>> 0")
            x = tf.pad(self.M2_0(inputs), [[0,0],[0,2]])
        elif x.argmax()==1:
            print(">>> 1")
            x = tf.pad(self.M2_1(inputs), [[0,0],[2,0]])
        else:
            assert False
        return x

M = EthnicityM_ensemble_eager() # 2020-01-07 19:18:08 初始化此模型要 71s
M.build((None,)+(IMAGE_SHAPE)+(3,))
M.summary()


# In[7]:


class EthnicityM_ensemble_tfn_cond(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        base_dir="/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/saved_models/{}"
#         self.M1 = tf.keras.models.load_model(base_dir.format("CauMon_vs_IndNeg_subclassed_inceptionv3__loss0.4735_acc0.8118_pb"))
#         self.M2_0 = tf.keras.models.load_model(base_dir.format("Cau_vs_Mon_subclassed_inceptionv3__loss0.4173_acc0.8937_pb"))
#         self.M2_1 = tf.keras.models.load_model(base_dir.format("Ind_vs_Neg_subclassed_inceptionv3__loss0.4566_acc0.8562_pb"))
        self.label=np.array(['Caucasoid','Mongoloid','Indian','Negroid'])
        self.M1 = M1
        self.M2_0 = M2
        self.M2_1 = M3
        self.M1.trainable=False
        self.M2_0.trainable=False
        self.M2_1.trainable = False
        tf.keras.backend.set_learning_phase(False)
    
    def func_(self, inp):
        inp_b = tf.expand_dims(inp,axis=0)
        p1 = tf.argmax(self.M1(inp_b), axis=1)
        p2_0 = tf.squeeze(tf.pad(self.M2_0(inp_b), [[0,0],[0,2]]))
        p2_1 = tf.squeeze(tf.pad(self.M2_1(inp_b), [[0,0],[2,0]]))
        return tf.cond(tf.equal(p1, 0), lambda:p2_0, lambda:p2_1)


    @tf.function()
    def call(self,inputs,training=False):
        x = tf.map_fn(self.func_, inputs, dtype=tf.float32)
        return x
    
    def custom_predict(self,input_batch):
        p1 = self.M1.predict(input_batch).argmax(axis=1)
        predictions=[]
        for idx,img in enumerate(input_batch):
            img_b = np.expand_dims(img, axis=0)
            if p1[idx] == 0:
                p2 = np.concatenate([np.squeeze(self.M2_0.predict(img_b)),[0.0,0.0]])
            elif p1[idx] == 1:
                p2 = np.concatenate([[0.0,0.0],np.squeeze(self.M2_1.predict(img_b))])
            else:
                assert False
            predictions.append(p2)
        return np.array(predictions)
    
    def custom_predict2(self, input_batch):
        p1 = self.M1.predict(input_batch)
        p2_0 = self.M2_0.predict(input_batch)
        p2_1 = self.M2_1.predict(input_batch)
        res = np.concatenate([p1,p2_0,p2_1], axis=1)
        return res
    
M = EthnicityM_ensemble_tfn_cond() # 2020-01-07 20:51:00 初始化此模型要 1m
M.build((None,)+(IMAGE_SHAPE)+(3,))
M.summary()


# ensemble模型检查是否正确

# In[10]:


imgArr_list_all=np.array([np.array(Image.open(p).resize(IMAGE_SHAPE))/255 for p in normal_flow_valid.filepaths])
inputs = imgArr_list_all[0:3]
print("M1:")
M.M1.predict(inputs)
M.M1.predict(inputs).argmax(axis=1)
print("M2_0:")
M.M2_0.predict(inputs)
print("M2_1:")
M.M2_1.predict(inputs)
print("M:")
M.predict(inputs)
M.custom_predict(inputs)
M.custom_predict2(inputs)


# 分析三个二分类模型的指标

# In[14]:


label = ['Caucasoid','Mongoloid','Indian','Negroid']
# label = ['Caucasoid','Mongoloid']
# label = ['Indian','Negroid']
# label = ['CauMon','IndNeg']
# augment_params={"horizontal_flip":True, "rotation_range":90}
ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255,**augment_params)
generic_params = dict(directory=sample_dir, target_size=IMAGE_SHAPE, batch_size=batch_size)

normal_flow_valid_CauMon_IndNeg = ig_normal.flow_from_directory(subset='validation', classes=["CauMon","IndNeg"],**generic_params)
normal_flow_valid_Cau_Mon = ig_normal.flow_from_directory(subset='validation', classes=["Caucasoid","Mongoloid"],**generic_params)
normal_flow_valid_Ind_Neg = ig_normal.flow_from_directory(subset='validation', classes=["Indian","Negroid"],**generic_params)

M.M1.evaluate(normal_flow_valid_CauMon_IndNeg)
M.M2_0.evaluate(normal_flow_valid_Cau_Mon)
M.M2_1.evaluate(normal_flow_valid_Ind_Neg)

# # auc
# y_label = np.array([label.index(i.split("/")[-2]) for i in sample_paths])
# y_pred = pred[:,1]

# calc=tf.keras.metrics.AUC()
# calc.update_state(y_label, y_pred)
# print("auc: ", calc.result().numpy())
# calc.reset_states()
# # acc
# print("acc at 0.5: ", tf.keras.metrics.binary_accuracy(y_label, y_pred, threshold=0.5).numpy())
# print(f"distribution of prediction=label[1] ({label[1]}): ")
# _ = plt.hist(y_pred, bins=400)


# In[6]:


IMAGE_SHAPE=(96,96)
ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255,**augment_params)
generic_params = dict(directory=sample_dir, target_size=IMAGE_SHAPE, batch_size=batch_size)

normal_flow_valid_CauMon_IndNeg = ig_normal.flow_from_directory(subset='validation', classes=["CauMon","IndNeg"],**generic_params)
normal_flow_valid_Cau_Mon = ig_normal.flow_from_directory(subset='validation', classes=["Caucasoid","Mongoloid"],**generic_params)
normal_flow_valid_Ind_Neg = ig_normal.flow_from_directory(subset='validation', classes=["Indian","Negroid"],**generic_params)

use_M="mobilenet_v2_075_96"
def get_sequential_model(rate=0.1, fine_tunning=False):
    feature_extractor_url = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(use_M)
    feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
    feat_layer.trainable = fine_tunning  # feature_vector的生成就不用训练了
    return tf.keras.Sequential([
      feat_layer,
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
M = get_sequential_model()
M.load_weights(tf.train.latest_checkpoint("/home/zhoutong/notebook_collection/tmp/CV_clf/ethnicity/by_script_SequentialM_CauMon_vs_IndNeg/ckpt_Sequential_Hub_fintunnig"))


# In[49]:


tf.keras.backend.set_learning_phase = False
normal_flow_valid_CauMon_IndNeg.next()[1].argmax(axis=1)
normal_flow_valid_CauMon_IndNeg.reset()
auc = tf.keras.metrics.AUC()
acc = tf.keras.metrics.CategoricalAccuracy()
for imgB,y in tqdm(normal_flow_valid_CauMon_IndNeg):
    p = M.predict(imgB)
    _ = auc.update_state(p[:,0],y[:,0]) # 反正是二分类，就拿idx=0当正样本吧
    _ = acc.update_state(p,y)
    if normal_flow_valid_CauMon_IndNeg.batch_index % len(normal_flow_valid_CauMon_IndNeg) == 0:
        break

acc.result().numpy(),auc.result().numpy()
acc.reset_states(),auc.reset_states()


# In[50]:


M.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])
M.evaluate(normal_flow_valid_CauMon_IndNeg)


# ## 示例测试(模型M)

# In[168]:


# M=model
M=M1 # Cau_Mon
M=M2 # Ind_Neg


# ### 测试整体分布

# In[67]:


M.build((None,)+IMAGE_SHAPE+(3,))


# #### 从apus独立验证目录验证NSFW

# In[68]:


NSFW_dir = "/home/zhoutong/notebook_collection/tmp/CV_clf/nsfw/apus_samples/NSFW"
SFW_dir = "/home/zhoutong/notebook_collection/tmp/CV_clf/nsfw/apus_samples/SFW"
def detect(fp_dir):
    sample_fp_list=[os.path.join(fp_dir,fp) for fp in os.listdir(fp_dir)]
    imgArr_list_all = [Image.open(p).resize(IMAGE_SHAPE) for p in sample_fp_list]
    imgArr_list_all = np.array([np.array(img)/255 for img in imgArr_list_all])
    imgArr_list_all.shape
    pred = M.predict(imgArr_list_all)
    print(pred.shape)
    return imgArr_list_all,pred

nsfw_imgArrs,nsfw_pred=detect(NSFW_dir)
sfw_imgArrs,sfw_pred=detect(SFW_dir)
print("NSFW")
print(np.unique([label[i] for i in nsfw_pred.argmax(axis=1)],return_counts=True))
print("SFW")
print(np.unique([label[i] for i in sfw_pred.argmax(axis=1)],return_counts=True))


# In[75]:


nsfw_err=[(idx,pred.max()) for idx,pred in enumerate(nsfw_pred) if pred.argmax() != label.index('NSFW')]
nsfw_err_img_top9=[(nsfw_imgArrs[idx],prob) for idx,prob in sorted(nsfw_err,key=lambda x:x[1],reverse=False)[:9]]

_ = ExqUtils.pltshow(imgArr_list= [img for img,prob in nsfw_err_img_top9],
                     info_list=[prob for img,prob in nsfw_err_img_top9])
plt.show()

sfw_err=[(idx,pred.max()) for idx,pred in enumerate(sfw_pred) if pred.argmax() != label.index('SFW')]
sfw_err_img_top9=[(sfw_imgArrs[idx],prob) for idx,prob in sorted(sfw_err,key=lambda x:x[1],reverse=True)[:9]]
if len(sfw_err_img_top9) == 1:
    img,prob=sfw_err_img_top9[0]
    _ = plt.imshow(img)
    plt.title=prob
    plt.show()
else:
    _ = ExqUtils.pltshow(imgArr_list= [img for img,prob in sfw_err_img_top9],
                         info_list=[prob for img,prob in sfw_err_img_top9])


# 从valid的flow检验

# In[90]:


# 注意 flow 默认是会shuffle的，要么手动把shuffle关掉( flow.shuffle=False ) 要么读一次通用
normal_flow_valid.reset()
normal_flow_valid.shuffle=False
sample_paths = normal_flow_valid.filepaths
print(">>> sample distribution:")
print(np.vstack(np.unique([i.split("/")[-2] for i in sample_paths],return_counts=True)).T)

M.build((None,)+IMAGE_SHAPE+(3,))
imgArr_list_all=np.array([np.array(Image.open(p).convert("RGB").resize(IMAGE_SHAPE))/255 for p in sample_paths])
pred = M.predict(imgArr_list_all)
print(">>> prediction distribution:")
print(np.vstack(np.unique([label[i] for i in np.argmax(pred,axis=1)],return_counts=True)).T)
print("整个prediction的max项的直方图:")
_ = plt.hist(np.max(pred,axis=1), bins=200)


# In[91]:


df = pd.DataFrame([{"label":sample_paths[idx].split("/")[-2],
                    "pred":label[int(np.argmax(i))],
                    "prob":np.max(i),
                    "pic":sample_paths[idx]} for idx,i in enumerate(pred)])
print(">>> 原始预测结果信息 head(3) 如下：")
df.head(3)
def wrap_stat(df):
    def func(dfg):
        item_list,cnt_list=np.unique(dfg['label'],return_counts=True)
        return {i:j for i,j in zip(item_list,cnt_list)}
    dfg = df.groupby("pred").apply(func).apply(pd.Series).fillna(0) # 这里的 pd.Series 不能写到func里
    dfg['pred_sum'] = sum([dfg[i] for i in dfg.columns])
    dfg=dfg.append(pd.Series({i:dfg[i].sum() for i in dfg.columns}, name="label_sum"), sort=False).fillna(0)
    # 注意：P和R不能直接复制（dfg["p"]=[...]） 例如先给dfg赋值了P，再计算R时取iteritems会多算一个P列导致KeyError
    P = [round(row[idx]/row['pred_sum'],4) if idx!='label_sum' else None for idx,row in dfg.iterrows()]
    R = pd.Series({idx: round(row[idx]/row['label_sum'],4) if idx!='pred_sum' else None for idx,row in dfg.iteritems()}, name="R")
    dfg['P'] = P
    dfg = dfg.append(R)
    dfg.columns=dfg.columns.set_names("label")
    return dfg
print(">>> 各类别预测结果分布如下：")
wrap_stat(df)
print(">>> 各类别预测结果（置信度>0.5）分布如下：")
wrap_stat(df.query("prob > 0.5"))
print(">>> 各类别预测结果（置信度>0.8）分布如下：")
wrap_stat(df.query("prob > 0.8"))
print(">>> 各类里预测结果的置信度分布？")
# imgArr_list = [np.array(Image.open(i)) for i in df.query("label=='Mongoloid' and pred != 'Mongoloid'")['pic']]
# fig,axes_list = plt.subplots(int(len(imgArr_list)**0.5), int(len(imgArr_list)**0.5)+1)
# for idx,img in enumerate(imgArr_list):
#     _ = axes_list.flatten()[idx].imshow(img)


# 具体数据分析

# In[92]:


if len(label) == 2:
    # auc
    y_label = np.array([label.index(i.split("/")[-2]) for i in sample_paths])
    y_pred = pred[:,1]

    calc=tf.keras.metrics.AUC()
    calc.update_state(y_label, y_pred)
    print("auc: ", calc.result().numpy())
    calc.reset_states()
    # acc
    print("acc at 0.5: ", tf.keras.metrics.binary_accuracy(y_label, y_pred, threshold=0.5).numpy())
    print(f"distribution of prediction=label[1] ({label[1]}): ")
    _ = plt.hist(y_pred, bins=400)


# ### 带图测试

# 取pred和label不同但置信度很高的样本看看

# In[93]:


# for check in ['Caucasoid']:
for check in label:
    print("*"*20,f"at {check}","*"*20)
    df_f = df.query(f"label=='{check}' and pred!='{check}'")
#     df_f = df.query(f"label=='{check}' and pred=='Indian'")
    #################
    # df的plot看下分布   
    #################
    _ = df_f['prob'].plot.hist()
    _ = df_f['prob'].plot.kde(bw_method=0.1) # bw_method 越小越拟合（可能会过拟合）
    print(f">>> label=={check} & pred!={check} 的prob分布及分位数如下:")
    plt.show()
    print(df_f['prob'].describe())
    
    ###############################
    # 按百分位抽头部的高置信度的错误样本
    ###############################
    hold = df_f['prob'].quantile(0.5)
    to_show=40
    print(f">>> 概率大于'{hold:.4f}' 总计有'{df_f[df_f['prob']>=hold].shape[0]}' top_{to_show}个如下")
    fig, axes = plt.subplots(int(to_show**0.5),int(to_show**0.5)+1, figsize=(15,15))
    fig.set_tight_layout(True)
    for idx,(df_idx, row) in enumerate(df_f.query(f"prob >={hold}").head(to_show).iterrows()):
        try:
            img = np.array(Image.open(row['pic']).resize((96,96)))
            axe = axes.flatten()[idx]
            axe.set_axis_off()
            _ = axe.imshow(img)
            info="[y]:{}\n[p]:{}={:.4f}\n{}".format(row['label'], row['pred'],row['prob'],row['pic'].split("/")[-1])
            _ = axe.set_title(info,size=8)
        except FileNotFoundError as  e:
          print(row['pic'],"not found")
    plt.show()


# 随机抽30张图预测看看

# In[59]:


sample = np.random.choice(sample_paths,30)
sample.shape
imgArr_list=np.array([np.array(Image.open(p).resize((IMAGE_SHAPE)))/255 for p in sample])
pred_raw = M.predict(imgArr_list)
pred = [label[int(np.argmax(i))] for i in pred_raw]

fig,axe_list = plt.subplots(6,5, figsize=(15,15))
fig.set_tight_layout(True)
for idx,img in enumerate(imgArr_list):
    axe = axe_list.flatten()[idx]
    axe.set_axis_off()
    _ = axe.imshow(img)
    y,p,prob = sample[idx].split("/")[-2], pred[idx],np.max(pred_raw[idx])
    _ = axe.text(x=0, y=axe.get_ylim()[0]+10, s="y:{} p:{}={:.4f}".format(y,p,prob), color="green" if y==p else "red")
    _ = axe.set_title(sample[idx].split("/")[-1],fontsize=7,color="green" if y==p else "red")


# ### 特征向量做聚类测试

# In[ ]:


import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.cluster import KMeans


# 初始化模型

# In[ ]:


class featureVectorURL:
    mobilenet="https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4"
    inception_resnet_v2 = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4"
    inception_v3="https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
    
feature_extractor_url = featureVectorURL.inception_v3
feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
feat_layer.trainable = False  # feature_vector的生成就不用训练了
featureM = tf.keras.Sequential([feat_layer])


# 加载fine-tuning后的权重

# In[ ]:


featureM.load_weights(tf.train.latest_checkpoint("./tmp/CV_clf/ethnicity/ckpt_Sequential_Hub_fintunnig/"))


# 直接使用M的feature层

# In[ ]:


featureM = tf.keras.Sequential([M.layers[0]])


# 训练集抽取特征向量

# In[ ]:


normal_flow_train.shuffle=False
normal_flow_train.reset()
total = normal_flow_train.samples//normal_flow_train.batch_size
total_feature_train = np.vstack([featureM.predict(normal_flow_train.next()[0]) for _ in tqdm(range(total))])


# 验证集抽取特征

# In[ ]:


normal_flow_valid.shuffle=False
normal_flow_valid.reset()
total = normal_flow_valid.samples//normal_flow_valid.batch_size
total_feature_valid = np.vstack([featureM.predict(normal_flow_valid.next()[0]) for _ in tqdm(range(total))])


# In[ ]:


# 统计某个图片和其他图片的cos相似度
res = []
for idx_v,f_v in tqdm(enumerate(total_feature_valid), total=len(total_feature_valid)):
    
    for idx_t,f_t in enumerate(total_feature_train):
        sim=cosine_similarity(np.expand_dims(f_t,axis=0),np.expand_dims(f_v,axis=0))[0,0]
        if sim > 0.8:
            res.append((idx_v,idx_t,sim))
            print("a",idx_v,idx_t,sim)

res_sorted = sorted(res, lambda x:x[-1])[::-1]


# 模型predict

# In[ ]:


trainX = []

normal_flow_valid.shuffle=False
normal_flow_valid.reset()
total = normal_flow_valid.samples//normal_flow_valid.batch_size
for idx,(img_batch,label_batch) in tqdm(enumerate(normal_flow_valid),desc="pre",total=total):
    if idx >= total:
        break
    feature_batch = featureM.predict(img_batch)
    label_batch = np.argmax(label_batch,axis=1)
    fp_batch = normal_flow_valid.filepaths[idx*normal_flow_valid.batch_size:(idx+1)*normal_flow_valid.batch_size]
    for i in range(normal_flow_valid.batch_size):
        trainX.append({"feature":feature_batch[i],"label":label_batch[i],"pic":fp_batch[i]})


# KMeans聚类

# In[ ]:


K = 2
kmeans = KMeans(n_clusters=K, random_state=0).fit([i['feature'] for i in trainX])
kmeans.cluster_centers_

for idx,center in enumerate(kmeans.cluster_centers_):
    _ = [dic.update({f"sim_to_c{idx}":cosine_similarity(np.expand_dims(center,axis=0),np.expand_dims(dic['feature'],axis=0))[0,0]}) for dic in tqdm(trainX)]

trainX[0]
Image.open(trainX[0]['pic']) # 某张图和c1、c3两个聚类中心的cos相似度都在0.8+ 看看图


# top相似召回演示

# In[ ]:


toShow=20

for idx in range(K):
    col = f'sim_to_c{idx}'
#     fig,axes=plt.subplots(int(toShow**0.5),int(toShow**0.5)+2,figsize=(8,8))
    fig,axes=plt.subplots(3,7,figsize=(10,7))
    fig.set_tight_layout(True)
    for idx, dic in enumerate(sorted(trainX, key=lambda x:x[col],reverse=True)[:toShow]):
        try:
            img=np.array(Image.open(dic['pic']).resize((96,96)))
            axe=axes.flatten()[idx]
            axe.set_axis_off()
            _=axe.imshow(img)
            _=axe.set_title(dic['label'],size=9)
#           _=axe.text(x=0,y=105,s=dic['pic'].split("/")[-1], size=8)
        except FileNotFoundError as e:
            print(dic['pic'], "not found")


# ### 请求TF Serving

# In[54]:


import requests
import json

fp="/home/zhoutong/notebook_collection/tmp/CV_clf/nsfw/samples/NSFW/dick_close_porn_114.jpg"
fp="/home/zhoutong/darknet/data/prepared_data/breast_close_porn_87.jpg"
imgPIL = Image.open(fp).resize((224,224))
imgPIL
imgArr=np.array(imgPIL)/255
# imgArr=np.zeros((224,224,3))

tf_serving_url = 'http://10.65.34.65:18051/v1/models/ethnicity:predict'
tf_serving_url = 'http://10.65.34.65:19054/v1/models/nsfw_bcnn:predict'
tf_serving_url = 'http://10.65.34.65:19053/v1/models/nsfw_obj:predict'
data = json.dumps({"signature_name": "serving_default", "instances": [imgArr.tolist()]})
headers = {"content-type": "application/json"}
json_response = requests.post(tf_serving_url, data=data, headers=headers)
# json_response.text
pred_batch = json.loads(json_response.text)['predictions']
pred_batch[0]['yolo_nms_3']
# pred[0]
# pred[0][::-1]


# In[58]:


output=['penis','vagina','breast']
pred=pred_batch[0]
boxes=pred['yolo_nms']
probs=pred['yolo_nms_1']
classes=pred['yolo_nms_2']
num=pred['yolo_nms_3']
for i in range(num):
    print(f"obj_idx:{i}\nobj_name: {class_names[int(classes[i])]}\nlabel: {classes[i]}\nprob: {prob[i]}")
    
label_prob_list=[(int(classes[i]),probs[i]) for i in range(num)]
obj_prob_hold=0.3
label_prob_list=[lp for lp in label_prob_list if lp[1]>=obj_prob_hold]
res = [{"label":l,"prob":p,"name":output[l]} for l,p in label_prob_list]
res


# docker部署的TF Serving
# 
# ```bash
# docker run -it --rm -p 8006:8501 \
#     -v "$(pwd)/tf_serving_models/ethnicity:/models/ethnicity" \
#     -e MODEL_NAME=ethnicity \
#     tensorflow/serving 
# ```

# In[65]:


import requests
label
tf_serving_url = 'http://localhost:8009/v1/models/ethnicity:predict'
tf_serving_url = 'http://10.65.34.65:18051/v1/models/ethnicity:predict'
# tf_serving_url = 'http://10.65.34.65:18052/v1/models/vectorize:predict'
to_show=3
sample_paths = np.random.choice(normal_flow_valid.filepaths,to_show)
print(">>> sample distribution:")
print(np.vstack(np.unique([i.split("/")[-2] for i in sample_paths],return_counts=True)).T)

imgArr_list_all=np.array([np.array(Image.open(p).resize(IMAGE_SHAPE))/255 for p in sample_paths])

# data = json.dumps({"signature_name": "serving_default", "instances": imgArr_list_all.tolist()})
data = json.dumps({"signature_name": "serving_default", "instances": [np.zeros((224,224,3)).tolist()]})
headers = {"content-type": "application/json"}
json_response = requests.post(tf_serving_url, data=data, headers=headers)
if json_response.status_code == 200:
    pred = np.array(json.loads(json_response.text)['predictions'])
else:
    print("ERROR: request fail",json_response.status_code)
    print(json_response.text)
    pred=None

    pred.shape
fig,axes = plt.subplots(int(to_show**0.5),int(to_show**0.5)+2,figsize=(15,15))
for idx,img in enumerate(imgArr_list_all[:30]):
    axe=axes.flatten()[idx]
    _=axe.set_axis_off()
    _=axe.imshow(img)
    y = sample_paths[idx].split("/")[-2]
    p = label[np.argmax(pred[idx])]
    t = "{} \n {}={}".format(y, p, np.max(pred[idx]))
    _ = axe.set_title(t,size=7,color="green" if y==p else "red")


# ### 加载url&人脸切割

# In[ ]:


import dlib
import urllib
from io import BytesIO
dlib_detector=dlib.get_frontal_face_detector()

def loadPIL_fromURL(url):
    url_response = urllib.request.urlopen(url)
    image = Image.open(BytesIO(url_response.read()))
    return image

def get_face(imgPIL, enlarge=0.1, target_size=(224,224)):
    imgArr = np.array(imgPIL)
    gray_img = np.array(imgPIL.convert("L"))
    rect_list = dlib_detector(gray_img, 1)
    face_img_list = []
    for rect in rect_list:
        (h, w) = (rect.height(), rect.width())
        (h, w) = (int(h * enlarge), int(w * enlarge))
        top = rect.top() - h if rect.top() - h > 0 else 0
        bottom = rect.bottom() + h if rect.bottom() + h < imgArr.shape[0] else imgArr.shape[0]
        left = rect.left() - h if rect.left() - h > 0 else 0
        right = rect.right() + h if rect.right() + h < imgArr.shape[1] else imgArr.shape[1]
        facePIL = Image.fromarray(imgArr[top:bottom, left:right]).resize(target_size)
        face_img_list.append(np.array(facePIL))
    return np.array(face_img_list)


# data
# 
# **IMPORTANT**:要记得rescale 255

# In[ ]:


url_list = ["https://p1.pstatp.com/large/9121000adae434816c34",
            "http://i2.bangqu.com/r2/news/20180706/304a5437534f474f3947.jpg",
            "http://cms-bucket.nosdn.127.net/catchpic/9/91/91ed88c3e725f8d355377dafd06d6565.jpg?imageView&thumbnail=550x0",
            "data:image/jpeg;base64,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",
            "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1576940296456&di=ffbb87c01ad30c3b619d81b4145565f0&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20181127%2Fe360946cdbc44a3fabd123ccabdc1d72.jpeg",
            "https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike92%2C5%2C5%2C92%2C30/sign=2ff1757977ec54e755e1124cd851f035/7a899e510fb30f244c691444c495d143ac4b0381.jpg",
            "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1576940435494&di=2bbbc24f85de8116079cc5f55f5f597e&imgtype=0&src=http%3A%2F%2Fn.sinaimg.cn%2Fsinacn%2Fw500h304%2F20180102%2F5c6b-fyqefvx1417506.jpg"
           ]


img_list = [loadPIL_fromURL(url) for url in url_list]
imgArr = np.vstack([get_face(img) for img in img_list])/255
imgArr.shape


# prediction by model M

# In[ ]:


pred = M.predict(imgArr)

total=imgArr.shape[0]
fig,axes = plt.subplots(int(total**0.5), int(total**0.5)+2, figsize=(8,6))
fig.set_tight_layout(True)
for idx,img in enumerate(imgArr):
    axe = axes.flatten()[idx]
    _ = axe.imshow(img)
    p = label[np.argmax(pred[idx])]
    prob = np.max(pred[idx])
    _ = axe.set_title(f"{p}={prob:.2f}")


# prediciton by request TF Serving

# In[ ]:


data = json.dumps({"signature_name": "serving_default", "instances": imgArr.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8009/v1/models/ethnicity:predict', data=data, headers=headers)
pred=json.loads(json_response.text)['predictions']

total=imgArr.shape[0]
fig,axes = plt.subplots(int(total**0.5), int(total**0.5)+2, figsize=(8,6))
fig.set_tight_layout(True)
for idx,img in enumerate(imgArr):
    axe = axes.flatten()[idx]
    _ = axe.imshow(img)
    p = label[np.argmax(pred[idx])]
    prob = np.max(pred[idx])
    _ = axe.set_title(f"{p}={prob:.2f}")


# In[ ]:


plt.imshow(imgArr[-1])
imgArr[-1].shape
plt.show()

# imgPIL = Image.open("./tmp/CV_clf/ethnicity/face/mtcn_182/Indian/india_woman_1.png").resize((224,224))
imgPIL = Image.open("/home/zhoutong/india_woman_1.png").convert("RGB").resize((224,224))

img = np.array(imgPIL)/255
plt.imshow(get_face(imgPIL)[0])
plt.show()

# p = M.predict(np.expand_dims(img, axis=0))
p = M.predict(get_face(imgPIL)/255)
img.shape
imgPIL
print("prob",p,"p",label[np.argmax(p,axis=1)[0]])

np.allclose(imgArr[-1], img, rtol=1e-4)


# ## 检查针对SubclassedM的各种测试

# In[16]:


from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# In[167]:


class TestM(tf.keras.Model):
    def __init__(self, scope:str="EthnicityM", rate:float=0.1, **kwargs):
        super().__init__(name=scope, **kwargs)
        self.L1 = tf.keras.layers.Dense(4,activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.L2 = tf.keras.layers.Dense(2,activation='softmax')
    def call(self, inp, training=False):
        x = self.L1(inp)
        x = self.dropout(x, training=training)
        x = self.L2(x)
        return x
    
    def build(self, input_shape, *args, **kwargs):
        self.inp_shape=input_shape[1:] # idx=0元素是batch_size，不取
        super().build(input_shape,*args,**kwargs)
    
    def summary(self, *args, **kwargs):
        x = tf.keras.Input(shape=self.inp_shape)
        tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()
        
class TestM_withVGG(tf.keras.Model):
    def __init__(self, scope:str="EthnicityM", rate:float=0.1, **kwargs):
        super().__init__(name=scope, **kwargs)
        vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        self.featM = tf.keras.Sequential(tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.L2 = tf.keras.layers.Dense(8,activation='softmax')
    def call(self, inp, training=False):
        x = self.featM(inp)
        x = self.dropout(x, training=training)
        x = self.L2(x)
        return x    


# ### dropout测试

# In[168]:


SHAPE=(5,)
M = TestM()
M.layers
M.build(input_shape=(None,)+SHAPE)
M.summary()


# In[181]:


tf.keras.backend.set_learning_phase(False)
predict0=[M.predict(tf.zeros((1,)+SHAPE))  for i in range(2)]
call0=[M(tf.zeros((1,)+SHAPE)).numpy()  for i in range(2)]
tf.keras.backend.set_learning_phase(True)
predict1=[M.predict(tf.zeros((1,)+SHAPE))  for i in range(2)]
call1=[M(tf.zeros((1,)+SHAPE)).numpy()  for i in range(2)]

print(">>> M.predict不论是否在训练都不出发dropout (完全一致): ")
predict0
predict1

print(">>> M call 在训练时调用会触发dropout: ")
call0
print("训练阶段触发dropout，两次结果不一致：")
call1


# ### 持久化策略的测试

# In[63]:


M = EthnicityM_App(input_shape_=IMAGE_SHAPE+(3,), num_classes=normal_flow_train.num_classes)
M.layers


# #### model.save_weights / load_weights

# In[64]:


fd = "/home/zhoutong/notebook_collection/tmp/CV_clf/tmp_ckpt_save_weights"
fp = os.path.join(fd,"ckpt_{e}")
M.save_weights(fp.format(e=2))
print(">>>> ckpt内部:")
print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(fd),tensor_name='',all_tensors='')
_ = M.load_weights(tf.train.latest_checkpoint(fd))
print("[success]")


# #### tf.train.Checkpoint / model.load_weights

# In[65]:


fp = "/home/zhoutong/notebook_collection/tmp/CV_clf/tmp_ckpt_tf.train.Checkpoint/"
ckpt = tf.train.Checkpoint(step=tf.Variable(1),model=M)
ckpt.save(fp)
print(">>>> ckpt内部:")
print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(fp),tensor_name='',all_tensors='')
_ = M.load_weights(tf.train.latest_checkpoint(fp))


# #### pb & serving

# In[51]:


# docker run -it --rm -p 8999:8501 \
#      -v "$PWD/tmp_pb:/models/ethnicity" \
#      -e MODEL_NAME=ethnicity \
#      tensorflow/serving
import requests
data = json.dumps({"signature_name": "serving_default", "instances": np.zeros((1,224,224,3)).tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post("http://localhost:8999/v1/models/ethnicity:predict", data=data, headers=headers)
json_response.text


# #### pb 

# In[46]:


fp = "/home/zhoutong/notebook_collection/tmp/CV_clf/tmp_pb"
# M.build((None,224,224,3))
M.predict(tf.zeros((1,224,224,3)))
tf.keras.models.save_model(model=M, filepath=fp, save_format="pb")
print("--- saved ---")
M1 = tf.keras.models.load_model(fp)
M1.predict(tf.zeros((1,224,224,3)))
print("[success]")


# ## 检查BN

# In[153]:


# 这里加不加 layers=tf.keras.layers 以下结果都不变
incep = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',include_top=False)
bns=[i for i in incep.layers if type(i).__name__ == "BatchNormalization"]
# 取InceptionV3的input_shape最大的一个BN层（这样均值方差的维度多方便看分布）
# bn0 = sorted(bns,key=lambda x: x.input_shape[-1])[-1]
bn0=bns[0]
print(">>> 直接以 「InceptionV3(weights='imagenet',include_top=False)」 方式加载，然后取input_shape最大的一个BN层实验")
print(">>> incep.trainable:{}, bn.trainable:{}, learning_phase:{}".format(incep.trainable, bn0.trainable, tf.keras.backend.learning_phase()))
print("    观察此时即使trainable是True，但是**learning_phase是False**")
print("    即bn在计算时其参数training收到的是False, mean、variance不会重新计算")
print(">>> 计算前\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

# 计算10次
_ = [bn0(tf.zeros((1,1,1,bn0.input_shape[-1]))) for i in range(10)]
print(">>> [注] 标记是否生效表示moving_mean moving_variance是否更新")
print(">>> [不生效]计算十次全零样本后\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

_ = [bn0(tf.zeros((1,1,1,bn0.input_shape[-1])),training=True) for i in range(10)]
print(">>> [生效] 手动指定bn的training=True再计算十次全零样本\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

tf.keras.backend.set_learning_phase=1
_ = [bn0(tf.zeros((1,1,1,bn0.input_shape[-1]))) for i in range(10)]
print(f">>> [不生效]手动指定learning_phase为True再计算十次全零样本 | tf.keras.backend.learning_phase()={tf.keras.backend.learning_phase()}\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))
print("     [注] tf2.0.0中set_learning_phase=1不生效（查看learning_phase仍为0）")


print(">>> incep.trainable:{}, bn.trainable:{}, learning_phase:{}".format(incep.trainable, bn0.trainable, tf.keras.backend.learning_phase()))
_ = [incep(tf.zeros((1,224,224,3))) for i in range(10)]
print(f">>> [不生效]incep模型 call十次全零样本\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

incep.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
_ = incep.fit(x=np.zeros((10,224,224,3)),y=np.ones((10,5,5,2048)))
print(f">>> [不生效]incep模型 fit一个batch(10个)全零样本\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

_ = incep.predict(x=np.zeros((10,224,224,3)))
print(f">>> [不生效]incep模型 predict十次全零样本\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

_ = incep.evaluate(x=np.zeros((10,224,224,3)),y=np.ones((10,5,5,2048)))
print(f">>> [不生效]incep模型 evaluate十次全零样本\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))

_ = [incep(tf.zeros((1,224,224,3)),training=True) for i in range(10)]
print(f">>> [生效]incep模型 call时指定training=True十次全零样本\n"+"mean,std of: "+" ".join(["[{}] {:.4f},{:.4f}".format(i.name.split("/")[-1], np.mean(i.numpy()), np.std(i.numpy())) for i in [bn0.beta, bn0.moving_mean, bn0.moving_variance]]))


# 为什么设置learning_phase不生效？

# In[119]:


tf.keras.backend.learning_phase()
tf.keras.backend.set_learning_phase=True
tf.keras.backend.learning_phase()
tf.keras.backend.set_learning_phase=1
tf.keras.backend.learning_phase()


# In[ ]:




