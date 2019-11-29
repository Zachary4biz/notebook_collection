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
import copy,os,sys,psutil
from collections import Counter,deque
import itertools
import os


# In[109]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[3]:


root_path = "/home/zhoutong/notebook_collection/data/cv/tmp/Image_samples"
cg_path = root_path+"/cg_background"
landscape = root_path+"/landscape"
taj = root_path + "/Taj_Mahal"
qutb = root_path + "/Qutb_Minar"
IMAGE_SHAPE = (224,224)


# In[4]:


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

# In[23]:


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

# In[6]:


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

# In[7]:


classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
clf = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])


# 加载headless model预测feature_vector

# In[8]:


feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))


# 注意:
# - clf是 `tf.keras.Sequential` 搭起来的
# - feat_layer是`hub.KerasLayer`直接做的一个`Layer`，所以得到的结果是一个`Tensor`

# In[9]:


image_batch.shape


# In[10]:


image_batch_part = image_batch[:16]
pred_res = clf.predict(image_batch_part)
print(f"clf的结果,shape: {pred_res.shape}, argmax: {np.argmax(pred_res, axis=1)}\n",pred_res)
feat_layer(image_batch_part)


# ## Model

# In[11]:


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

# In[13]:


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

# In[18]:


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

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib


# In[62]:


get_ipython().run_line_magic('pinfo', 'df.dropna')


# In[64]:


df = pd.read_csv("/home/zhoutong/res.csv")
df['id'] = df['id'].astype(int)
df.count()
print("去掉na")
df = df.dropna()
df.count()
df.head(3)


# In[68]:


ctr = df['ctr'].to_numpy()
print("各百分位对应的ctr:")
[(i,np.percentile(ctr,i)) for i in [5,15,25,50,75,95]]
print("取上四分位为正样本，下四分位为负样本")
neg,pos = df.query("ctr<0.022"),df.query("ctr>0.068")
print(f"正样本计数: {pos.size}, 负样本计数: {neg.size}")
_ = plt.hist(ctr[np.logical_and(ctr<0.13, ctr>0)],bins=300)
plt.show()
print("去掉极端值后分布如上，中位数:",np.percentile(ctr[np.logical_and(ctr<0.13, ctr>0)], 50))
print("均值:",np.mean(ctr[np.logical_and(ctr<0.13, ctr>0)]))


# In[73]:


neg['banner_url'].size


# In[72]:


np.unique(neg['banner_url']).size


# In[77]:


sample_dir="./tmp/auto_score"
print("下载图片至本地: ",sample_dir)

def download_pos(url):
    path = sample_dir+"/pos/"+url.split("/")[-1].split("?")[0]
    urllib.request.urlretrieve(url,path)
    
def download_neg(url):
    path = sample_dir+"/neg/"+url.split("/")[-1].split("?")[0]
    urllib.request.urlretrieve(url,path)
    
from multiprocessing import Pool
p = Pool(4)
iter_to_run = p.imap(download_neg, neg['banner_url'])
_ = list(tqdm(iter_to_run,total=neg['banner_url'].size,desc="download neg:"))

iter_to_run = p.imap(download_pos, pos['banner_url'])
_ = list(tqdm(iter_to_run,total=pos['banner_url'].size,desc="download pos:"))


# In[78]:


IMAGE_SHAPE = (224,224)
batch_size = 1
validation_ratio=0.3
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


# In[83]:


feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4" #@param {type:"string"}
feat_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))
feat_layer.trainable = False  # feature_vector的生成就不用训练了
model = tf.keras.Sequential([
  feat_layer,
  tf.keras.layers.Dense(normal_flow_train.num_classes, activation='softmax')
])
model.summary()
pred = model(image_batch)
pred
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])


# In[84]:


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


# In[85]:


batch_stats_callback = CollectBatchStats()
history = model.fit_generator(normal_flow_train, epochs=4,
                              steps_per_epoch = normal_flow_train.samples//40,
                              validation_data = normal_flow_valid,
                              validation_steps = normal_flow_valid.samples ,
                              callbacks = [batch_stats_callback])


# In[90]:


dir(history)


# In[97]:


history.params
history.history


# ## 模型保存

# In[ ]:


# 使用 `checkpoint_path` 格式保存权重
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# model.save_weights(checkpoint_path.format(epoch=0))
# model.save("mymodel.h5")


# In[98]:


model.save("./tmp/auto_score/models/model_bysave.h5")


# In[88]:


tf.keras.models.save_model(
    model=model,
    filepath="./tmp/auto_score/models/model_1",
    overwrite=True,
    include_optimizer=True,
    save_format="h5",
    signatures=None,
)


# ## 模型加载

# In[103]:


bysave = tf.keras.models.load_model("./tmp/auto_score/models/model_bysave.h5", custom_objects={'KerasLayer':hub.KerasLayer})


# In[102]:


bytfkeras = tf.keras.models.load_model("./tmp/auto_score/models/model_1", custom_objects={'KerasLayer':hub.KerasLayer})


# In[105]:


type(bysave)
type(bytfkeras)


# In[117]:


pos_path="/home/zhoutong/notebook_collection/tmp/auto_score/pos/{}"
neg_path="/home/zhoutong/notebook_collection/tmp/auto_score/neg/{}"

img = Image.open(pos_path.format("2060713010_1570648202589.jpg")).resize((224,224))
img


# In[118]:


bysave.build((None,224,224,3))
bysave.predict(np.expand_dims(np.array(img),0))


# In[ ]:




