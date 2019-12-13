#!/usr/bin/env python
# coding: utf-8

# 基于Incepiton的迁移学习，用 tf、keras、tfhub 实现
# 
# 参照 [官网tutorials](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub#download_the_headless_model)

# In[1]:


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


# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


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

# In[3]:


tf.config.experimental.list_physical_devices()


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


# In[ ]:


sess_conf = tf.ConfigProto()
sess_conf.gpu_options.allow_growth = True  # 允许GPU渐进占用
sess_conf.allow_soft_placement = True  # 把不适合GPU的放到CPU上跑

g_graph = tf.Graph()
g_sess = tf.Session(graph=g_graph, config=sess_conf)


# # 正式流程

# In[ ]:


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(root_path, classes=['Taj_Mahal','Qutb_Minar'], target_size=IMAGE_SHAPE)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

# 重新加载一下iter
image_data = image_generator.flow_from_directory(root_path, classes=['Taj_Mahal','Qutb_Minar'])


# # 测试

# ## 合并多个generator

# 使用tf封装的`ImageDataGenerator`从目录里直接读取数据，按子目录来分类别
# 
# 这里尝试了合并多个 generator | 参考 [SO的回答](https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs)
# - 注意测试了`itertools.chain`是不行的，会一直循环第一个`generator`的结果
# - `concat` 方法可行但是没有保留`flow_from_directory`得到的类`DirectoryIterator`，一些方法如`next()`, `num_class`不能用了，`batch_size`也需要更新为`n倍`，这些都只能用新的变量单独保存
# 
# ```python
# def concat(*iterables):
#     while True:
#         data = [i.next() for i in iterables]
#         yield np.concatenate([i[0] for i in data], axis=0), np.concatenate([i[1] for i in data], axis=0)
# 
# to_merge = [image_data_aug1,image_data_aug2,image_data_normal]
# train_data = concat(*to_merge)
# num_classes = image_data_aug1.num_classes
# batch_size = len(to_merge) * batch_size
# ```
# - 还有一种方法是继承Keras的`Sequence`类，但是这方法似乎也没有保留`DirectoryIterator`的那些属性，没有尝试 上述的 [SO回答](https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs) 中有这种方案

# 合并三个 generator，各自代表不同的augmentaion —— 水平翻转&缩放、旋转&明暗、正常

# In[ ]:


datagen_args1 = dict(horizontal_flip=True,zoom_range=[0.1,0.2])
datagen_args2 = dict(rotation_range=90, brightness_range=[0.3,0.5])
batch_size = 90
image_generator_aug1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,**datagen_args1)
image_generator_aug2 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,**datagen_args2)
image_generator_normal = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data_aug1 = image_generator_aug1.flow_from_directory(root_path, classes=['Taj_Mahal','Qutb_Minar'], target_size=IMAGE_SHAPE, batch_size=batch_size)
image_data_aug2 = image_generator_aug2.flow_from_directory(root_path, classes=['Taj_Mahal','Qutb_Minar'], target_size=IMAGE_SHAPE, batch_size=batch_size)
image_data_normal = image_generator_normal.flow_from_directory(root_path, classes=['Taj_Mahal','Qutb_Minar'], target_size=IMAGE_SHAPE, batch_size=batch_size)

def concat(*iterables):
    while True:
        data = [i.next() for i in iterables]
        yield np.concatenate([i[0] for i in data], axis=0), np.concatenate([i[1] for i in data], axis=0)

to_merge = [image_data_aug1,image_data_aug2,image_data_normal]
train_data = concat(*to_merge)
num_classes = image_data_aug1.num_classes
batch_size = len(to_merge) * batch_size
print(f">>> merge {len(to_merge)} 个iter后的batch_size为: {batch_size}")
print(">>> 如下显示加了 [旋转、反转] 等augmentation的训练集（合并了多个generator）")
for i in range(199*3//batch_size+1+1):
    pics, label_batch = train_data.__next__()
    if i == 0:
        print(" 独立演示train_data里的第一项")
        print(" Image batch shape: ", pics.shape)
        print(" Label batch shape: ", label_batch.shape)
        print("~~"*15)
    print(pics.shape)
    if i == 199*3//batch_size:
        print(">>> 已经消费完所有数据，后面从头开始从generator里获取数据")
    r = int(len(pics) ** 0.5)
    c = len(pics) // r + 1
    fig,axes_arr = plt.subplots(r,c)
    _ = [ax.set_axis_off() for ax in axes_arr.ravel()]
    for idx, pic in enumerate(pics):
        axes = axes_arr[idx//c, idx % c]
        axes.set_axis_off()
        _ = axes.imshow(pic)

image_batch, label_batch = train_data.__next__()  # 随便拿一个出来当image_batch给后面测试用


# ## 带有train test/validation 的Generator
# 基本流程是
# - 初始化`ImageDataGenerator`时提供`validation_split`参数
# - 然后获取flow时（例如`flow_from_directory`）使用`subset`来标记是取训练集还是测试集

# In[ ]:


batch_size = 90
validation_ratio=0.2
ig_aug1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,horizontal_flip=True,zoom_range=[0.1,0.2])
ig_aug2 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,rotation_range=90, brightness_range=[0.3,0.5])
generic_params = dict(directory=root_path, classes=['Taj_Mahal','Qutb_Minar'], target_size=IMAGE_SHAPE, batch_size=batch_size)
augflow1 = ig_aug1.flow_from_directory(**generic_params)
augflow2 = ig_aug2.flow_from_directory(**generic_params)
# 一般用没有augmentation的数据做验证集
generic_params = dict(directory=root_path, classes=['Taj_Mahal','Qutb_Minar'], target_size=IMAGE_SHAPE, batch_size=batch_size)
ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255)
normal_flow_train = ig_normal.flow_from_directory(subset='training', **generic_params)
normal_flow_valid = ig_normal.flow_from_directory(subset='validation', **generic_params)


def concat(*iterables):
    while True:
        data = [i.next() for i in iterables]
        yield np.concatenate([i[0] for i in data], axis=0), np.concatenate([i[1] for i in data], axis=0)

to_merge = [augflow1,augflow2,normal_flow_train]
train_data = concat(*to_merge)
num_classes = augflow1.num_classes
samples = sum(i.samples for i in to_merge)
batch_size = len(to_merge) * batch_size
print(f">>> merge {len(to_merge)} 个iter后的batch_size为: {batch_size}")
print(">>> 如下显示加了 [旋转、反转] 等augmentation的训练集（合并了多个generator）")
for i in range(samples//batch_size+1+2): # 多循环两轮
    pics, label_batch = train_data.__next__()
    if i == 0:
        print(" 独立演示train_data里的第一项")
        print(" 注意后续的shape会比较特别是因为，ig_normal分了20%做验证集，所以会比另外两个没有分validation的提前消耗完")
        print(" 假设batch_size=90,它在第二次取90个时就消耗完了,只取到了160-90=70个,而另外两个数据集还能取到90个,总计就是70+90*2=250个")
        print(" Image batch shape: ", pics.shape)
        print(" Label batch shape: ", label_batch.shape)
        print("~~"*15)
    print(pics.shape)
    if i == samples//batch_size:
        print(">>> 已经消费完所有数据，下一次会从头开始从generator里获取数据")
    r = int(len(pics) ** 0.5)
    c = len(pics) // r + 1
    fig,axes_arr = plt.subplots(r,c)
    _ = [ax.set_axis_off() for ax in axes_arr.ravel()]
    for idx, pic in enumerate(pics):
        axes = axes_arr[idx//c, idx % c]
        axes.set_axis_off()
        _ = axes.imshow(pic)

image_batch, label_batch = train_data.__next__()  # 随便拿一个出来当image_batch给后面测试用
print(f">>> 消费完后从头拿到的数据shape是: {image_batch.shape}")


# 加载classification model预测分类

# In[ ]:


classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
clf = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])


# 加载headless model预测feature_vector

# In[ ]:


feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))


# 注意:
# - clf是 `tf.keras.Sequential` 搭起来的
# - feat_layer是`hub.KerasLayer`直接做的一个`Layer`，所以得到的结果是一个`Tensor`

# In[ ]:


image_batch.shape


# In[ ]:


image_batch_part = image_batch[:16]
pred_res = clf.predict(image_batch_part)
print(f"clf的结果,shape: {pred_res.shape}, argmax: {np.argmax(pred_res, axis=1)}\n",pred_res)
feat_layer(image_batch_part)


# ## Model

# In[ ]:


feat_layer.trainable = False  # feature_vector的生成就不用训练了
model = tf.keras.Sequential([
  feat_layer,
  tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.summary()
pred = model(image_batch)
pred
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])


# ## CallBack

# In[ ]:


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


# 上面用`concat`方式的结果，要注意`validation_step`
# - 如果用`normal_flow_valid.samples // batch_size`注意是否为0

# ## Fit

# In[ ]:


steps_per_epoch = np.ceil(samples//batch_size)
batch_stats_callback = CollectBatchStats()
history = model.fit_generator(normal_flow_train, epochs=4,
                              steps_per_epoch = normal_flow_train.samples//40,
                              validation_data = normal_flow_valid,
                              validation_steps = normal_flow_valid.samples ,
                              callbacks = [batch_stats_callback])


# In[ ]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
train_data_dir = "tf_files/codoon_photos"
validation_data_dir = "tf_files/codoon_photos"
nb_train_samples = 4125
nb_validation_samples = 466 
batch_size = 16
epochs = 50

model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the all layers.
for layer in model.layers[:]:
    layer.trainable = False

# Adding custom Layer
# We only add
x = model.output
x = Flatten()(x)
# Adding even more custom layers
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30)

test_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size = (img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
  validation_data_dir,
  target_size = (img_height, img_width),
  class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("resnet50_retrain.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
model_final.fit_generator(
  train_generator,
  samples_per_epoch = nb_train_samples,
  epochs = epochs,
  validation_data = validation_generator,
  nb_val_samples = nb_validation_samples,
  callbacks = [checkpoint, early])


# # 机器打分

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib


# ## 数据准备

# - **准备「train」「validation」数据**

# In[ ]:


print("\n","--"*15,"原始数据准备","--"*15,"\n")
df = pd.read_csv("/home/zhoutong/res.csv")
df['id'] = df['id'].astype(int)
df.count()
print("去掉na")
df = df.dropna()
df.count()
df.head(3)

print("\n","--"*15,"取样本","--"*15,"\n")
ctr = df['ctr'].to_numpy()
print("各百分位对应的ctr:")
[(i,np.percentile(ctr,i)) for i in [5,15,25,50,75,95]]
print("取上四分位为正样本，下四分位为负样本")
neg,pos = df.query("ctr<0.022"),df.query("ctr>0.068")
print(f"正样本计数: {pos.shape}, 负样本计数: {neg.shape}")
_ = plt.hist(ctr[np.logical_and(ctr<0.13, ctr>0)],bins=300)
plt.show()
print("去掉极端值后分布如上，中位数:",np.percentile(ctr[np.logical_and(ctr<0.13, ctr>0)], 50))
print("均值:",np.mean(ctr[np.logical_and(ctr<0.13, ctr>0)]))


# - **获取「verify」数据**

# In[ ]:


print("\n","--"*15,"原始数据准备","--"*15,"\n")
df_v = pd.read_csv("/home/zhoutong/res_11_25.csv")
df_v['id'] = df_v['id'].astype(int)
df_v.count()
print("去掉na")
df_v = df_v.dropna()
df_v.count()
df_v['fileName'] = df_v['banner_url'].apply(lambda url:url.split("/")[-1].split("?")[0])
df_v.head(3)

# print("\n","--"*15,"取样本","--"*15,"\n")
# ctr = df['ctr'].to_numpy()
# print("各百分位对应的ctr:")
# [(i,np.percentile(ctr,i)) for i in [5,15,25,50,75,95]]
# print("取上四分位为正样本，下四分位为负样本")
# neg,pos = df.query("ctr<0.022"),df.query("ctr>0.068")
# print(f"正样本计数: {pos.shape}, 负样本计数: {neg.shape}")
# _ = plt.hist(ctr[np.logical_and(ctr<0.13, ctr>0)],bins=300)
# plt.show()
# print("去掉极端值后分布如上，中位数:",np.percentile(ctr[np.logical_and(ctr<0.13, ctr>0)], 50))
# print("均值:",np.mean(ctr[np.logical_and(ctr<0.13, ctr>0)]))


# 下载图片

# In[ ]:


sample_dir="./tmp/auto_score"
print("下载图片至本地: ", sample_dir)

def download(url, subdir):
    path = os.path.join(sample_dir, subdir, url.split("/")[-1].split("?")[0])
    urllib.request.urlretrieve(url, path)

    
from functools import partial
download_pos = partial(download, subdir="pos")
download_neg = partial(download, subdir="neg")
download_verify = partial(download, subdir="verify")
from multiprocessing import Pool
p = Pool(12)
iter_to_run = p.imap(download_verify, df_v['banner_url'])
_ = list(tqdm(iter_to_run, total=df_v['banner_url'].shape[0], desc="download verify_pics:"))

# iter_to_run = p.imap(download_neg, neg['banner_url'])
# _ = list(tqdm(iter_to_run,total=neg['banner_url'].size,desc="download neg:"))

# iter_to_run = p.imap(download_pos, pos['banner_url'])
# _ = list(tqdm(iter_to_run,total=pos['banner_url'].size,desc="download pos:"))


# ## Train/Test数据

# In[ ]:


IMAGE_SHAPE = (224,224)
batch_size = 32
validation_ratio=0.1
sample_dir="./tmp/auto_score"
generic_params = dict(directory=sample_dir, classes=['pos','neg'], target_size=IMAGE_SHAPE, batch_size=batch_size)
ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255)
normal_flow_train = ig_normal.flow_from_directory(subset='training', **generic_params)
normal_flow_valid = ig_normal.flow_from_directory(subset='validation', **generic_params)

for image_batch, label_batch in normal_flow_train:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

# 重新加载一下iter
normal_flow_train = ig_normal.flow_from_directory(subset='training', **generic_params)


# ## 模型

# In[ ]:


feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4" #@param {type:"string"}
feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))
feat_layer.trainable = False  # feature_vector的生成就不用训练了
model = tf.keras.Sequential([
  feat_layer,
  tf.keras.layers.Dense(normal_flow_train.num_classes, activation='softmax')
])
model.summary()
# pred = model(image_batch)
# pred
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])


# In[ ]:


class HistoryCB(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
        self.logs_dict={}

    def on_train_batch_end(self, batch, logs=None):
        print("cococococco at ",batch)
        self.logs_dict.update({batch:logs})


# ## 训练

# In[ ]:


tbd_cb=tf.keras.callbacks.TensorBoard(log_dir='./tmp/auto_score/tensorboard_logs',
                                      histogram_freq=0,write_graph=False,update_freq='batch',
                                      profile_batch=0)

checkpoint_path = "./tmp/auto_score/ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
ckpt_cb=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True)

es_db = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# progress_db = tf.keras.callbacks.ProgbarLogger(count_mode='steps')

custom_db = HistoryCB()

history = model.fit_generator(normal_flow_train, epochs=15,
                              steps_per_epoch = normal_flow_train.samples // normal_flow_train.batch_size,
                              validation_data = normal_flow_valid,
#                               validation_steps = normal_flow_valid.samples,
                              verbose=1,
                              callbacks=[])


# In[ ]:


history.params
history.history


# ## 模型保存

# In[ ]:


# 使用 `checkpoint_path` 格式保存权重
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# model.save_weights(checkpoint_path.format(epoch=0))
# model.save("mymodel.h5")


# In[ ]:


model.save("./tmp/auto_score/models/model_bysave.h5")


# In[ ]:


tf.keras.models.save_model(
    model=model,
    filepath="./tmp/auto_score/models/model_1",
    overwrite=True,
    include_optimizer=True,
    save_format="h5",
    signatures=None,
)


# In[ ]:


import requests
get_ipython().run_line_magic('pinfo', 'requests.get')


# In[ ]:


pos_path="/home/zhoutong/notebook_collection/tmp/CV_auto_score/pos/{}"
neg_path="/home/zhoutong/notebook_collection/tmp/CV_auto_score/neg/{}"
verify_path = "/home/zhoutong/notebook_collection/tmp/CV_auto_score/verify/{}"
r=16
total = pd.concat([df_v.head(r),df_v[190:190+r],df_v[500:500+r],df_v.tail(r)])
total.head(10)

total_res =[]
for idx,row in total.iterrows():
    total_res.append((row['banner_url'],row['ctr'],np.array(Image.open(verify_path.format(row['fileName'])).resize((224,224)))))
fig, axe_list=plt.subplots(8,8,figsize=(15,16))

for idx,(url,ctr,img) in enumerate(total_res):
    axe = axe_list.flatten()[idx]
    axe.set_axis_off()
    obj_text=",".join([i['obj']+":"+str(i['cnt']) for i in json.loads(requests.get("http://10.65.34.65:8004/obj",params={"img_url":url,"id":-1}).text)['result']])
#     obj_text="asdf"
    _ = axe.imshow(img)
    _ = axe.text(x=0, y=axe.get_ylim()[0]+50, s="{:4f}\n{}".format(ctr,obj_text))


# ## 模型加载

# In[ ]:


bysave = tf.keras.models.load_model("./tmp/auto_score/models/model_bysave.h5", custom_objects={'KerasLayer':hub.KerasLayer})
bysave.build((None,224,224,3))
type(bysave)
print(">>> test: ")
type(bysave)


# In[ ]:


bysave.predict(np.array([np.array(img) for img in img_pos]))
bysave.predict(np.array([np.array(img) for img in img_neg]))


# In[ ]:


bytfkeras = tf.keras.models.load_model("./tmp/auto_score/models/model_1", custom_objects={'KerasLayer':hub.KerasLayer})
bytfkeras.build((None,224,224,3))
print(">>> test: ")
type(bytfkeras)
bytfkeras.predict(np.expand_dims(np.array(img),0))


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

# In[5]:


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


# ### 人种数据

# In[ ]:


IMAGE_SHAPE = (96,96)
batch_size = 1
validation_ratio=0.1
sample_dir = "./tmp/CV_clf/ethnicity/face/mtcn_182"
label = ['Indian','Negroid','Caucasoid','Mongoloid']
generic_params = dict(directory=sample_dir, classes=label, target_size=IMAGE_SHAPE, batch_size=batch_size)
ig_normal = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_ratio,rescale=1/255)
normal_flow_train = ig_normal.flow_from_directory(subset='training', **generic_params)
normal_flow_valid = ig_normal.flow_from_directory(subset='validation', **generic_params)

for image_batch, label_batch in normal_flow_train:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

# 重新加载一下iter
normal_flow_train.reset()
print(">>> 训练集分布：")
print(np.vstack(np.unique([i.split("/")[-2] for i in normal_flow_train.filepaths],return_counts=True)).T)
print(">>> 测试集分布：")
print(np.vstack(np.unique([i.split("/")[-2] for i in normal_flow_valid.filepaths],return_counts=True)).T)


# ## 临时 | 用SVM试试
# 多分类SVM还是不好弄

# In[3]:


import sklearn
from sklearn.svm import SVC


# In[6]:


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4" #@param {type:"string"}
feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
feat_layer.trainable = False  # feature_vector的生成就不用训练了
featureM = tf.keras.Sequential([feat_layer])


# In[14]:


# 得到向量 存起来
trainX = []
trainY = []
total = normal_flow_train.samples//normal_flow_train.batch_size
for idx,(img_batch,label_batch) in tqdm(enumerate(normal_flow_train),desc="pre",total=total):
    if idx >= total:
        break
    trainX.append(featureM.predict(img_batch))
    trainY.append(label_batch)

trainX = np.array(trainX)
trainX = np.reshape(trainX, (-1,trainX.shape[-1]))
trainY = np.array(trainY)
trainY = np.reshape(trainY, (-1,trainY.shape[-1]))

np.save("./tmp/CV_clf/ethnicity/trainX_feature.npy",trainX)
np.save("./tmp/CV_clf/ethnicity/trainY.npy",trainY)

validX = []
validY = []
total = normal_flow_valid.samples//normal_flow_valid.batch_size
for idx,(img_batch,label_batch) in tqdm(enumerate(normal_flow_valid),desc="pre",total=total):
    if idx >= total:
        break
    validX.append(featureM.predict(img_batch))
    validY.append(label_batch)

validX = np.array(validX)
validX = np.reshape(validX, (-1,validX.shape[-1]))
validY = np.array(validY)
validY = np.reshape(validY, (-1,validY.shape[-1]))


np.save("./tmp/CV_clf/ethnicity/validX_feature.npy",validX)
np.save("./tmp/CV_clf/ethnicity/validY.npy",validY)


# In[33]:


trainX = np.load("./tmp/CV_clf/ethnicity/trainX_feature.npy")
trainY = np.load("./tmp/CV_clf/ethnicity/trainY.npy")
validX = np.load("./tmp/CV_clf/ethnicity/validX_feature.npy")
validY = np.load("./tmp/CV_clf/ethnicity/validY.npy")


# In[20]:


clf = SVC()
validY = [1 if np.all(l==[1,0,0,0,0]) else 0 for l in validY]
clf.fit(validX, validY)


# In[24]:


clf.support_vectors_
clf.support_vectors_.shape
clf.n_support_


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

checkpoint_dir = "./tmp/CV_clf/ethnicity/ckpt_Subclass_KApp"


# ### Subclass + Hub

# In[ ]:


class EthnicityM_Hub(tf.keras.Model):
    def __init__(self,num_classes:int, scope:str="EthnicityM", **kwargs):
        super().__init__(name="EthnicityM", **kwargs)
        # 参考自： https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
        feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4"
        self.feature_model = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
        self.feature_model.trainable=False
        # 全连接
        self.prediction_layer = tf.keras.layers.Dense(num_classes)
        
    def call(self, inp, training=False):
        x = self.feature_model(inp)
        x = self.prediction_layer(x)
        return x

model=EthnicityM_Hub(normal_flow_train.num_classes)
model.build((None,96,96,3))
model.summary()

checkpoint_dir = "./tmp/CV_clf/ethnicity/ckpt_Subclass_Hub"


# ### Sequential + Hub

# In[27]:


def get_sequential_model():
    feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4" #@param {type:"string"}
    feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE+(3,))
    feat_layer.trainable = False  # feature_vector的生成就不用训练了
    return tf.keras.Sequential([
      feat_layer,
      tf.keras.layers.Dense(normal_flow_train.num_classes, activation='softmax')
    ])
model = get_sequential_model()
model.summary()

checkpoint_dir = "./tmp/CV_clf/ethnicity/ckpt_Sequential_Hub"


# ## 训练

# ### 自己写循环

# tensorboard

# In[ ]:


# tensorboard写入第一组图片
summary_writer = tf.summary.create_file_writer("./tmp/CV_clf/ethnicity/tensorboard")
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


# 准备变量

# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
ce_loss_fn = tf.keras.losses.categorical_crossentropy
# mean_calc = tf.keras.metrics.Mean()  # 这个Mean是累积的，也就是说到了第100batch它计算的实际上是0~100batch的均值
acc_fn = tf.keras.metrics.categorical_accuracy


# ckpt
# - ckpt还需要知道opt和model? （推测应该是不指定无法resotre）
# 
# 看看官方手册
# - [完整ckpt & ckpt-manager 的示例](https://www.tensorflow.org/guide/checkpoint)
# - [checkpoint类](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)

# In[ ]:


checkpoint_path = os.path.join(checkpoint_dir,"ckpt_{epoch}")
# ckpt = tf.train.Checkpoint(opt=opt,model=model)
# status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
# checkpoint.save(file_prefix=checkpoint_prefix)
ckpt = tf.train.Checkpoint(step=tf.Variable(1))

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)


# 训练loop

# In[ ]:


normal_flow_train.reset()
step_per_epoch = normal_flow_train.samples // normal_flow_train.batch_size
best_valid_acc = 0.0
for e in range(15):
    for step, (image_batch, label_batch) in tqdm(enumerate(normal_flow_train), desc=f"Epoch: {e}", total=step_per_epoch):
        if step>= step_per_epoch:
            break
        with tf.GradientTape() as tape:
            pred_batch = model(image_batch)
            loss_batch = ce_loss_fn(label_batch,pred_batch)
            acc_batch = acc_fn(label_batch,pred_batch)
        gradients = tape.gradient(loss_batch, model.trainable_variables)
        _ = opt.apply_gradients(zip(gradients, model.trainable_variables))
        loss,acc = np.mean(loss_batch),np.mean(acc_batch)
        with summary_writer.as_default(): 
            _ = tf.summary.scalar('train_loss', loss, step=step+e*step_per_epoch)
            _ = tf.summary.scalar('train_acc', acc, step=step+e*step_per_epoch)
        
    _ = ckpt.step.assign_add(1)
    val_pred = model.predict_generator(normal_flow_valid)
    val_label = tf.one_hot(normal_flow_valid.labels,depth=normal_flow_valid.num_classes)
    val_loss = np.mean(ce_loss_fn(val_label, val_pred).numpy())
    val_acc = np.mean(acc_fn(val_label, val_pred).numpy())
    with summary_writer.as_default(): 
        _ = tf.summary.scalar('val_loss', val_loss, step=step+e*step_per_epoch)
        _ = tf.summary.scalar('val_acc', val_acc, step=step+e*step_per_epoch)
    print(f'[e]:{e} [step]:{step} [loss]:{loss:.4f} [acc]:{acc:.4f} [val_loss]:{val_loss:.4f} [val_acc]:{val_acc:.4f}')
    if val_acc > best_valid_acc:
        best_valid_acc = val_acc
        save_path = ckpt_manager.save()
        print(f"acc improved [from]:{best_valid_acc:.4f} [to]:{val_acc:.4f}.\n[ckpt-path]: {save_path}")
    else:
        print(f"acc NOT improved [from]:{best_valid_acc:.4f}")


# ### 使用Keras的 fit

# In[ ]:


checkpoint_path = os.path.join(checkpoint_dir,"ckpt_{epoch}")
ckpt_cb=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                           save_weights_only=True, 
                                           verbose=1, save_best_only=True)

# 使用Adam就用不上这个cb了
def decay(epoch):
    if  epoch <= 4:
        return 0.045
    elif 4 < epoch and epoch <= 10:
        return 1e-3
    else:
        return 1e-5
lr_cb = tf.keras.callbacks.LearningRateScheduler(decay)

tbd_cb=tf.keras.callbacks.TensorBoard(log_dir='./tmp/CV_clf/ethnicity/tensorboard',
                                      histogram_freq=1,write_graph=False,update_freq='batch',
                                      profile_batch=1)


# In[ ]:


model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=['acc'])
history = model.fit_generator(normal_flow_train, epochs=20,
                              steps_per_epoch = normal_flow_train.samples // normal_flow_train.batch_size,
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

# In[ ]:


# model.save("./tmp/CV_clf/ethnicity/saved_models/sequential_mobilenetv2_acc0.5895_loss1.02758.h5")
model.save("./tmp/CV_clf/ethnicity/saved_models/sequential_inceptionresnetv2_acc0.5895_loss1.02758.h5")


# ## 模型 加载

# ### load ckpt

# In[ ]:


M = get_sequential_model()
# M = EthnicityM_Hub(normal_flow_train.num_classes)
print(f"loading from ckpt: {checkpoint_dir}")
_ = M.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
M.count_params()
M


# ### load saved_model

# In[ ]:


M = tf.keras.models.load_model("./tmp/CV_clf/ethnicity/saved_models/sequential_inceptionresnetv2_acc0.5895_loss1.02758.h5",custom_objects={"KerasLayer":hub.KerasLayer})
M.count_params()
M


# ## 示例测试

# In[ ]:


# 注意 flow 默认是会shuffle的，要么手动把shuffle关掉( flow.shuffle=False ) 要么读一次通用
sample_paths = normal_flow_valid.filepaths
print(">>> sample distribution:")
print(np.vstack(np.unique([i.split("/")[-2] for i in sample_paths],return_counts=True)).T)

imgArr_list_all=np.array([np.array(Image.open(p).resize((96,96)))/255 for p in sample_paths])
pred = M.predict(imgArr_list_all)
print(">>> prediction distribution:")
print(np.vstack(np.unique([label[i] for i in np.argmax(pred,axis=1)],return_counts=True)).T)


# ### 测试整体分布

# In[ ]:


df.sort_values('prob')


# In[ ]:


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


# ### 带图测试

# 取pred和label不同但置信度很高的样本看看

# In[ ]:


check="Negroid"
df_f = df.query(f"label=='{check}' and pred!='{check}'")

_ = df_f['prob'].plot.hist()
# bw_method 越小越拟合（可能会过拟合）
_ = df_f['prob'].plot.kde(bw_method=0.1)
print(f">>> label=={check} & pred!={check} 的prob分布及分位数如下:")
plt.show()
print(df_f['prob'].describe())

hold = df_f['prob'].quantile(0.85)
print(f">>> 概率大于'{hold:.4f}' 总计有'{df_f[df_f['prob']>=hold].shape[0]}'个 show20如下")

to_show=20
fig, axes = plt.subplots(int(to_show**0.5),int(to_show**0.5)+1, figsize=(15,15))
# for idx,(df_idx, row) in enumerate(df_f.query(f"prob >={hold}").head(to_show).iterrows()):
for idx,(df_idx, row) in enumerate(df_f.query(f"prob <0.5").head(to_show).iterrows()):
    img = np.array(Image.open(row['pic']).resize((96,96)))
    axe = axes.flatten()[idx]
    axe.set_axis_off()
    _ = axe.imshow(img)
    info="[y]:{}\n[p]:{}={:.4f}".format(row['label'], row['pred'],row['prob'])
    _ = axe.set_title(info)


# 随机抽30张图预测看看

# In[ ]:


sample = np.random.choice(sample_paths,30)
sample.shape
imgArr_list=np.array([np.array(Image.open(p).resize((96,96)))/255 for p in sample])
pred_raw = M.predict(imgArr_list)
pred = [label[int(np.argmax(i))] for i in pred_raw]
pred_prob = [np.max(i) for i in pred_raw]
fig,axe_list = plt.subplots(6,5, figsize=(15,20))
for idx,img in enumerate(imgArr_list):
    axe = axe_list.flatten()[idx]
    axe.set_axis_off()
    _ = axe.imshow(img)
    y,p,prob = sample[idx].split("/")[-2], pred[idx],pred_prob[idx]
    _ = axe.text(x=0, y=axe.get_ylim()[0]+10, s="y:{} p:{}={:.4f}".format(y,p,prob), color="green" if y==p else "red")
    _ = axe.set_title(sample[idx].split("/")[-1],fontsize=7,color="green" if y==p else "red")


# In[ ]:




