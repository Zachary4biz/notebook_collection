#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter,deque


# In[60]:


import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # 禁用GPU
from PIL import Image


# In[80]:


class Samples:
    blue_birds = [
        "http://www.kedo.gov.cn/upload/resources/image/2017/04/24/150703.png",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3Wm23HuKYuKMiSo9U_UAFDYc1_ccodPS9PMNrOWesI3lAE0bF&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQythkV6vmH4FnVuiJFkPAnj-_iAca42bMf1eZQDEGKEnO5zMzC&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThwIfzyp-Rv5zYM0fwPmoM5k1f9eW3ETYuPcL8j2I0TuG0tdb5&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzzG3lfGmXZJN2OPQvxRTTLqwMvVIaHd-BVrC88FQvtpUuMidR&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOT85WhH3PJ7VNA64bC3bmm_wH3UUt33xHOaT8Mc7unj1F_p0l&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSG5gPEotCi2WMBj5lyh8ftdRwjiySyViFeu-eB24bIKW_SFOrP&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTq6-juHAdZNKHHDdBNZ5fWGywrVxhLGFEdLE6mcSo5WdsPNw9_&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAoKVoo9HZFqWWfuJDKGGmPybhj99pn2kVcplJFZbtyrE9csFI&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-i6UBlc4TbDH85S4LYh9rYvdE-F9eZX0Azr0n4ySSRxBtPPnd&s",
        "http://img.boqiicdn.com/Data/BK/A/1908/9/imagick71411565338658_y.jpg"
    ]
    cartoon = [
        "https://png.pngtree.com/png-clipart/20190617/original/pngtree-ancient-characters-q-version-moe-illustration-png-image_3834249.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWzxDGvkniSg2uYGB1lY06cicXM2tsjrYDRE1GwoS9rBJQHPP7&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYs1FUZTewRyaceTlP8nAEX0jI9zLM2Z_0o9zdDT26FkINsI9r&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScY_5WL6lFFP510ZZBlKbbUTi1oxw79XZOU3KwrB5-clkZ4o7J&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_g4up-VwBIP0UESG3qEAXvhng_-rQhKbBPs1zlFvh28bjCXdm&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLzZPSSo0OXvSptcIZkiq_8HkHiFr7gOcYZHfnjeq0jtBD4_dh&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4OMEvJnADiAV0wST0dTdaRJHIFSwrG9L-U47lakXpq1wiwekx&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmhOowW42bpCwh9hofbJjsINh3MSQm_c4ZgNI5mnXKT33u81uL&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGI2N9hko1pymx_EoarfVXxBwz7fBETPwcLDD_qG1Htq65zJ_K&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQbwmMandg0e1VZl_BGW8hJlMD5bItcgcRrUX1cQtwEl-mMqvTp&s"
    ]
    white_cat = [
        "https://pic2.zhimg.com/v2-0f46e56eb41906c6969540478bae5184_1200x500.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLd7WnZCNZV0da1Urw9wDg2HtlfK6PUBYKSw4lKllmGPHe-AHU&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjKcBZjcyToc0rhlWTEIF8p49jh05munfP3q61t5o_Hj6Zygit&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRf6ZjHx5CvBOSAGSWdKnT-pNl2uItbUSV181LLFAb46Y6c7CFq&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFDarJmcFqwn6wNFOKE971bzZMz6jRFbuOUbsic_6sx7F-O6M0&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTqkTyLYyP_4B1SftKYaEn-gGlxHpBUySNxqKkIlCiX09n2MLX7&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSt78JI12oeg6haidCNAUhSAhbJXfKfQpw6o98G45H-SSqPVlIK&s"
    ]
    pyramid = [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXb-J88mYMC6Zy4hsGv315xANFBy-cyXG0iIwMERSVoB1iP9Ej&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThcdt1nV1YElMCPhQVVKLo3cb7ZdE_gZmjMfVP0vX7dnDqNMSd&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREBvRy6kXUKtdZFfDdD_s1UZd8LorkudYfgO0QzMF89UDXx_ih&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQO_6IOgZtWwCxtj0q1BnDdZwYd_ScDUcbXQ5VPeIhk6ZoSNbqm&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQkCpsOKFGhJATu6ucg3ayeGtS9rWkpTejHS0GJfn8cncG_k-Z&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfauaep8uZMRxzaGXp7M_vhMB4s52wGKdk8DHW_dcFpG7j4Jp6&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRDBamzIg-jEbblkDNGXkWTE7lblIlpOnodJ9oMZYkU86rE1PKv&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLnAwZ-PIQ-T8Dgl1Wvk6YKTQW77aqLRkMSJZnSEK4IAB6dhdF&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRkzthzjRc5iacNhmof9Jfywav7cnpsVdWXhw_8jzW7EZeow_10&s",
    ]


# # TF Slim 加载
# - 这个基本被TF团队弃用了，TF2.0也不再支持
# 
# 训练好的ckpt文件: https://github.com/tensorflow/models/blob/master/research/slim/README.md#pre-trained-models
# 

# In[21]:


from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib import slim


# In[5]:


def PreprocessImage(image, central_fraction=0.875, img_size=[128,128]):
    """Load and preprocess an image.

    Args:
      image: a tf.string tensor with an JPEG-encoded image.
      central_fraction: do a central crop with the specified
        fraction of image covered.
    Returns:
      An ops.Tensor that produces the preprocessed image.
    """

    # Decode Jpeg data and convert to float.
    image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)

    image = tf.image.central_crop(image, central_fraction=central_fraction)
    # Make into a 4D tensor by setting a 'batch size' of 1.
    image = tf.expand_dims(image, [0])
    image = tf.image.resize_bilinear(image,
                                     img_size,
                                     align_corners=False)

    # Center the image about 128.0 (which is done during training) and normalize.
    image = tf.multiply(image, 1.0 / 127.5)
    return tf.subtract(image, 1.0)


# In[23]:


g = tf.Graph()
sess = tf.Session(graph=g)
with g.as_default():
    input_image = tf.placeholder(tf.string)
    processed_image = PreprocessImage(input_image, img_size=[256,256]) # img_size不能小于128x128
    processed_image.shape

#     with slim.arg_scope([slim.conv2d, slim.fully_connected], 
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params={'is_training': False, 'updates_collections': None}):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = vgg.vgg_16(processed_image,is_training=False)
#         logits, end_points = inception.inception_v3(processed_image, 
#                                                     num_classes=1000, 
#                                                     is_training=False)
        saver = tf.train.Saver()


# In[ ]:


f1=sess.run(img_feature,feed_dict={input_image:"./tmp/CBIR/blue_bird/img2.png"})
f2=sess.run(img_feature,feed_dict={input_image:"./tmp/CBIR/blue_bird/150703.png"})
f3sess.run(img_feature,feed_dict={input_image:"./tmp/CBIR/blue_bird/cat.png"})


# inception_v3输入图片的size是固定的(299,299)，格式遵循此[common image input](https://www.tensorflow.org/hub/common_signatures/images#input)

# In[57]:


import tensorflow_hub as hub
from tensorflow.keras import layers
feature_vec_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
f_vec = tf.keras.Sequential([hub.KerasLayer(feature_vec_url, output_shape=[2048],trainable=False)])
f_vec.get_config()
writer = tf.summary.FileWriter("./tmp/CBIR/keras_tensorboard/", tf.keras.backend.get_session().graph)


# In[78]:


def get_img(url,IMAGE_SHAPE = (299,299)):
    img_file = tf.keras.utils.get_file('image.jpg',url)
    img = Image.open(img_file).resize(IMAGE_SHAPE)
    img_arr = np.array(img)/255.0  # 一般tfhub里的模型图片输入都要归一化到[0,1]
    return img_arr

grace_hopper_batch = get_img('https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper_batch = grace_hopper_batch[np.newaxis,:]  # 在axis=0上增加一列，即增加一个batch维度

Samples.blue_birds


# In[73]:


f_vec.predict(grace_hopper_batch)


# In[43]:


help(clf)


# In[45]:


clf.get_config()


# In[ ]:




