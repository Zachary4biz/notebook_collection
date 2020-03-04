from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys
from collections import Counter,deque
import itertools
import os
import time
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import pandas as pd
import subprocess
import dlib


# IMAGE_SHAPE = (96,96)
IMAGE_SHAPE = (224,224)
IMAGE_SHAPE = (299,299)
batch_size = 32
validation_ratio=0.2
sample_dir = "./tmp/CV_clf/ethnicity/face/dlib_frontal_0.25"
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



model.featM.trainable = False
print(f"trainable:{model.featM.trainable}")
bn=[i for i in model.featM.layers if type(i).__name__ =="BatchNormalization"][0]
print(">>> fit前")
for i in [(i.name, np.mean(i.numpy()),np.std(i.numpy())) for i in bn.variables]:
    print(i)
model.featM.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])
model.featM.fit(x=normal_flow_train.next()[0],y=np.ones((normal_flow_train.batch_size,1,1,2048)))
print(">>> fit后")
for i in [(i.name, np.mean(i.numpy()),np.std(i.numpy())) for i in bn.variables]:
    print(i)

