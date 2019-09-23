#!/usr/bin/env python
# coding: utf-8

# In[9]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[35]:


import tensorflow as tf
import datetime
import gensim
import numpy as np
import pandas as pd
from zac_pyutils.ExqUtils import zprint
from zac_pyutils import ExqUtils
from collections import deque
from tqdm.auto import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
# from gensim.models.wrappers import FastText
import fasttext
import json
import re
import inspect


# # Config

# In[45]:


class TrainingConfig(object):
    epoches = 5
    batchSize = 128
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001
    minWordCnt = 5


class ModelConfig(object):
    numFilters = 64

    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.001

class Config(object):
    job = "taste"
    basePath = "/home/zhoutong/nlp/data/textcnn"
    dataSource = basePath + "/labeled_timeliness_region_taste_emotion_sample.json"
    # dataSource = dataSource + ".sample_h10k"
    summaryDir = basePath+"/summary"
    cnnmodelPath_pb = basePath + "/textcnn_model_pb"
    cnnmodelPath_ckpt = basePath+"/textcnn_model_ckpt/model.ckpt"

    weDim = 300
    ft_modelPath = basePath + '/cc.en.300.bin'


    padSize = 16
    pad = '<PAD>'
    pad_initV = np.zeros(weDim)
    unk = '<UNK>'
    unk_initV = np.random.randn(weDim)

    # numClasses = 4  # 二分类设置为1，多分类设置为类别的数目
    numClasses_dict = {"taste":4,"timeliness":9,"emotion":3}
    numClasses = numClasses_dict[job]  # 二分类设置为1，多分类设置为类别的数目

    testRatio = 0.2  # 测试集的比例

    training = TrainingConfig()

    model = ModelConfig()

class Utils():
    # 清理符号
    @staticmethod
    def clean_punctuation(inp_text):
        res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./“”]", r' ', inp_text)
        res = re.sub(r"\\u200[Bb]", r' ', res)
        res = re.sub(r"\n+", r" ", res)
        res = re.sub(r"\s+", " ", res)
        return res.strip()
    @staticmethod
    def pad_list(inp_list,width,pad_const):
        if len(inp_list) >= width:
            return inp_list[:width]
        else:
            return inp_list+[pad_const]*(width-len(inp_list))
    
config = Config()


# In[49]:


data.trainReviews.shape[0]
527480
config.training.batchSize
527480//config.training.batchSize


# In[75]:


# fail
import timeout_decorator
@timeout_decorator.timeout(seconds=3, use_signals=False, exception_message="timeout")
def get_input(res_inpt):
    res_inpt=input("是否重新生成persist数据？(y/n)")

res = None
try:
    get_input(res)
except Exception as e:
    res = "N"
print(res)


# In[ ]:


# fail
class StoppableThread(Thread):
    def __init__(self,target,args,timeout=1):
        super(Thread,self).__init__()
        self.timeout = timeout
        self.stopped = False
        self.t = Thread(target=target,args=args)
        self.t.setDaemon(True)

    def start(self):
        self.t.start()
        while not self.stopped:
            self.t.join(self.timeout)
        print("thread stopped.")
    
    def stop(self):
        self.stopped=True

res = "n"
t1 = StoppableThread(target=get_input,args=(res,))
t1.start()
time.sleep(3)
t1.terminate()
print(res)


# In[ ]:


class StoppableThread(Thread):
    def __init__(self,target,args,timeout=1):
        super(Thread,self).__init__()
        self.timeout = timeout
        self.stopped = False
        self.t = Thread(target=target,args=args)
        self.t.setDaemon(True)

    def start(self):
        self.t.start()
        while not self.stopped:
            self.t.join(self.timeout)
        print("thread stopped.")
    
    def stop(self):
        self.stopped=True


def func():
    pass
    
    
t1 = StoppableThread(target=func)
t1.start()
for i in range(5):
    time.sleep(1)
    print(i)
t1.terminate()


# In[1]:


print("abc")


# In[12]:


from threading import Thread
import time
import queue

class StoppableThread(Thread):
    class TimeoutException(Exception):
        pass
    
    def __init__(self,target,args=None,time_limit=1,delta=0.05):
        super(Thread,self).__init__()
        self.delta = delta
        self.stopped = False
        if args is None:
            self.t = Thread(target=target)
        else:
            self.t = Thread(target=target,args=args)
        self.t.setDaemon(True)
        
        self.timing_thread = Thread(target=self.timing,args=(time_limit,))
        self.timing_thread.setDaemon(True)
    
    def timing(self,timeout):
        time.sleep(timeout)
        self.stopped=True
        
    def start(self):
        self.t.start()
        self.timing_thread.start()
        while not self.stopped:
            self.t.join(self.delta)
            time.sleep(0.05)
        raise TimeoutException("thread timeout")

    
    def stop(self):
        self.stopped=True


q = queue.Queue()
def func(inp):
    print("input: ",inp)
    for i in range(3):
        time.sleep(1)
        print(i)
    inp += 1
    print("processed: ",inp)
    q.put(inp)

to_inp = 2
t1 = StoppableThread(target=func,args=(to_inp,), time_limit=1)
t1.start()
print("to_inp: ",to_inp)
print("result in queue:",q.queue[0])


# In[19]:


len(q.queue)


# In[14]:


print("result in queue:",q.queue[0])


# In[9]:


print(q.queue[0])
dir(q)


# In[39]:


zprint("各数据路径：")
print("basePath路径：{}\n样本数据来源: {}\nsummary目录：{}\n".format(config.basePath,config.dataSource,config.summaryDir))
zprint("模型参数如下：")
for k,v in inspect.getmembers(config.model):
    if not k.startswith("_"):
        print(k,v)
zprint("训练参数如下：")
for k,v in inspect.getmembers(config.training):
    if not k.startswith("_"):
        print(k,v)


# # TextCNN

# In[3]:


class TextCNN(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.padSize], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        # 有三种size的filter，2, 3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filterShape = [filterSize, config.weDim, 1, config.model.numFilters]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                convRes = tf.nn.conv2d(
                    input=self.embeddedWordsExpanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(convRes, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.padSize - filterSize + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)

        # 池化后的维度不变，按照最后的维度channel来concat
        self.hPool = tf.concat(pooledOutputs, 3)

        # 摊平成二维的数据输入到全连接层
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

        # dropout
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

            print(self.predictions)

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss


# In[4]:


class MetricUtils():
    """
    定义各类性能指标
    """
    @staticmethod
    def mean(item: list) -> float:
        """
        计算列表中元素的平均值
        :param item: 列表对象
        :return:
        """
        res = sum(item) / len(item) if len(item) > 0 else 0
        return res

    @staticmethod
    def accuracy(pred_y, true_y):
        """
        计算二类和多类的准确率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]
        corr = 0
        for i in range(len(pred_y)):
            if pred_y[i] == true_y[i]:
                corr += 1
        acc = corr / len(pred_y) if len(pred_y) > 0 else 0
        return acc

    @staticmethod
    def binary_precision(pred_y, true_y, positive=1):
        """
        二类的精确率计算
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param positive: 正例的索引表示
        :return:
        """
        corr = 0
        pred_corr = 0
        for i in range(len(pred_y)):
            if pred_y[i] == positive:
                pred_corr += 1
                if pred_y[i] == true_y[i]:
                    corr += 1

        prec = corr / pred_corr if pred_corr > 0 else 0
        return prec

    @staticmethod
    def binary_recall(pred_y, true_y, positive=1):
        """
        二类的召回率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param positive: 正例的索引表示
        :return:
        """
        corr = 0
        true_corr = 0
        for i in range(len(pred_y)):
            if true_y[i] == positive:
                true_corr += 1
                if pred_y[i] == true_y[i]:
                    corr += 1

        rec = corr / true_corr if true_corr > 0 else 0
        return rec

    @staticmethod
    def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
        """
        二类的f beta值
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param beta: beta值
        :param positive: 正例的索引表示
        :return:
        """
        precision = MetricUtils.binary_precision(pred_y, true_y, positive)
        recall = MetricUtils.binary_recall(pred_y, true_y, positive)
        try:
            f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        except:
            f_b = 0
        return f_b

    @staticmethod
    def multi_precision(pred_y, true_y, labels):
        """
        多类的精确率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param labels: 标签列表
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]

        precisions = [MetricUtils.binary_precision(pred_y, true_y, label) for label in labels]
        prec = MetricUtils.mean(precisions)
        return prec

    @staticmethod
    def multi_recall(pred_y, true_y, labels):
        """
        多类的召回率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param labels: 标签列表
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]

        recalls = [MetricUtils.binary_recall(pred_y, true_y, label) for label in labels]
        rec = MetricUtils.mean(recalls)
        return rec

    @staticmethod
    def multi_f_beta(pred_y, true_y, labels, beta=1.0):
        """
        多类的f beta值
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :param labels: 标签列表
        :param beta: beta值
        :return:
        """
        if isinstance(pred_y[0], list):
            pred_y = [item[0] for item in pred_y]

        f_betas = [MetricUtils.binary_f_beta(pred_y, true_y, beta, label) for label in labels]
        f_beta = MetricUtils.mean(f_betas)
        return f_beta

    @staticmethod
    def get_binary_metrics(pred_y, true_y, f_beta=1.0):
        """
        得到二分类的性能指标
        :param pred_y:
        :param true_y:
        :param f_beta:
        :return:
        """
        acc = MetricUtils.accuracy(pred_y, true_y)
        recall = MetricUtils.binary_recall(pred_y, true_y)
        precision = MetricUtils.binary_precision(pred_y, true_y)
        f_beta = MetricUtils.binary_f_beta(pred_y, true_y, f_beta)
        return acc, recall, precision, f_beta

    @staticmethod
    def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
        """
        得到多分类的性能指标
        :param pred_y:
        :param true_y:
        :param labels:
        :param f_beta:
        :return:
        """
        acc = MetricUtils.accuracy(pred_y, true_y)
        recall = MetricUtils.multi_recall(pred_y, true_y, labels)
        precision = MetricUtils.multi_precision(pred_y, true_y, labels)
        f_beta = MetricUtils.multi_f_beta(pred_y, true_y, labels, f_beta)
        return acc, recall, precision, f_beta


# # Class Dataset

# In[5]:


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource

        self.testRatio = config.testRatio
        self._we_fp = config.basePath+"/wordEmbeddingInfo"  # \t分割 词,idx,embedding
        self._tokens_arr_fp = config.basePath+"/tokens_arr.npy"
        self._labels_arr_fp = config.basePath+"/labels_arr.npy"
        self._emb_arr_fp = config.basePath+"/emb_arr.npy"
        self.ft_modelPath = config.ft_modelPath
        self.ft_model = None


        self.trainReviews = np.array([])
        self.trainLabels = np.array([])

        self.evalReviews = np.array([])
        self.evalLabels = np.array([])

        self.token2idx = {}
        self.wordEmbedding = None
        self.labelSet = []
        self.totalWordCnt = 0

    def _readData(self, filePath):
        f_iter = ExqUtils.load_file_as_iter(filePath)
        tokens_list = deque()
        label_list = deque()
        zprint("loading data from: "+filePath)
        for l in tqdm(f_iter,desc="readData"):
            info = json.loads(l)
            text,label = info['title'],info[self.config.job]
            tokens = Utils.pad_list(Utils.clean_punctuation(text).split(" "),width=self.config.padSize,pad_const=self.config.pad)
            tokens_list.append(tokens)
            label_list.append(label)
        return np.array(tokens_list), np.array(label_list)

    def _initStopWord(self, stopWordPath):
        with open(stopWordPath, "r") as fr:
            self._stopWordSet = set(fr.read().splitlines())

    def _initFasttextModel(self):
        if self.ft_model is None:
            self.ft_model = fasttext.load_model(self.ft_modelPath)

    def _tokens2idx(self,tokens_arr):
        tokensSet = set(np.unique(tokens_arr))

        pass

    def dataGen_persist(self):
        """
                初始化训练集和验证集
                """
        zprint("init fasttext model")
        self._initFasttextModel()

        # 初始化数据集
        tokens_arr, label_arr = self._readData(self._dataSource)
        self.labelSet = set(np.unique(label_arr))
        tokensSet = set(np.unique(tokens_arr))

        self.totalWordCnt = len(tokensSet)
        wordEmb = np.zeros(shape=[self.totalWordCnt, self.ft_model.get_dimension()])
        # (idx,token,emb)保存到文本文件
        zprint("预测词向量总计: {} , 词向量存入文件: {}".format(self.totalWordCnt, self._we_fp))
        with open(self._we_fp, "w") as fw:
            # 加上 <PAD> 和 <UNK> 及其初始化
            for idx, token in tqdm(enumerate(tokensSet), total=self.totalWordCnt,desc="tokensSet"):
                if token == self.config.pad:
                    emb = self.config.pad_initV
                elif token == self.config.unk:
                    emb = self.config.unk_initV
                else:
                    emb = self.ft_model[token]
                self.token2idx.update({token:idx})
                wordEmb[idx] = emb
                fw.writelines(str(idx) + "\t" + token + "\t" + ",".join([str(i) for i in list(emb)]) + "\n")

        # tokens变为idx保存为npy
        zprint("tokens映射为索引保存到npy")
        tokensIdx_arr = np.zeros_like(tokens_arr, dtype=np.int64)
        for i,tokens in enumerate(tokens_arr):
            for j,token in enumerate(tokens):
                tokensIdx_arr[i][j] = self.token2idx[token]
        np.save(self._tokens_arr_fp,tokensIdx_arr)

        zprint("labels保存到npy")
        np.save(self._labels_arr_fp,label_arr)

        zprint("idx对应的emb保存到npy")
        np.save(self._emb_arr_fp,wordEmb)

    def loadData(self):
        self.wordEmbedding = np.load(self._emb_arr_fp)

        tokensIdx_arr = np.load(self._tokens_arr_fp)
        label_arr = np.load(self._labels_arr_fp)
        self.labelSet = set(np.unique(label_arr))
        # 初始化训练集和测试集
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.testRatio, random_state=2019)
        train_idx, test_idx = list(sss.split(np.zeros(label_arr.shape[0]), label_arr))[0]

        self.trainReviews = tokensIdx_arr[train_idx]
        self.trainLabels = label_arr[train_idx]

        self.evalReviews = tokensIdx_arr[test_idx]
        self.evalLabels = label_arr[test_idx]


# # Produce Data

# In[11]:


data = Dataset(config)
# data.dataGen_persist()
data.loadData()
print("train data shape: {}".format(data.trainReviews.shape))
print("train label shape: {}".format(data.trainLabels.shape))
print("eval data shape: {}".format(data.evalReviews.shape))
print("eval data shape: {}".format(data.evalLabels.shape))
print("wordEmbedding info file: {}".format(data._we_fp))


# In[12]:


trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
labelList = data.labelSet


# In[13]:


trainReviews
trainLabels


# # 开始构建计算图

# In[11]:


evalLabels.shape
batchY = np.array(evalLabels[10: 100], dtype="float32")
batchY.shape


# In[15]:


def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x,y = x[perm],y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# In[55]:


tf.reset_default_graph() 


# In[56]:


self = cnn
# 定义模型的输入
self.inputX = tf.placeholder(tf.int32, [None, config.padSize], name="inputX")
self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
# 定义l2损失
l2Loss = tf.constant(0.0)
# 词嵌入层
with tf.name_scope("embedding"):
    # 利用预训练的词向量初始化词嵌入矩阵
    self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
    # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
    self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
    # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
    self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)
# 创建卷积和池化层
pooledOutputs = []
# 有三种size的filter，2, 3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
for i, filterSize in enumerate(config.model.filterSizes):
    with tf.name_scope("conv-maxpool-%s" % filterSize):
        # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
        # 初始化权重矩阵和偏置
        filterShape = [filterSize, config.weDim, 1, config.model.numFilters]
        W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
        convRes = tf.nn.conv2d(
            input=self.embeddedWordsExpanded,
            filter=W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # relu函数的非线性映射
        h = tf.nn.relu(tf.nn.bias_add(convRes, b), name="relu")
        # 池化层，最大池化，池化是对卷积后的序列取一个最大值
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, config.padSize - filterSize + 1, 1, 1],
            # ksize shape: [batch, height, width, channels]
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
# 得到CNN网络的输出长度
numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)
# 池化后的维度不变，按照最后的维度channel来concat
self.hPool = tf.concat(pooledOutputs, 3)
# 摊平成二维的数据输入到全连接层
self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])
# dropout
with tf.name_scope("dropout"):
    self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)
# 全连接层的输出
with tf.name_scope("output"):
    outputW = tf.get_variable(
        "outputW",
        shape=[numFiltersTotal, config.numClasses],
        initializer=tf.contrib.layers.xavier_initializer())
    outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
    l2Loss += tf.nn.l2_loss(outputW)
    l2Loss += tf.nn.l2_loss(outputB)
    self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
    if config.numClasses == 1:
        self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
    elif config.numClasses > 1:
        self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
    print(self.predictions)
# 计算二元交叉熵损失
with tf.name_scope("loss"):
    if config.numClasses == 1:
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                         labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                        dtype=tf.float32))
    elif config.numClasses > 1:
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
    self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss


# In[82]:


self.logits.shape
self.inputY.shape
self.hDrop.shape,outputW.shape,outputB.shape
self.hPoolFlat.shape
inpX.shape,inpY.shape
filterShape
convRes.shape
pooled.shape
pooledOutputs
numFiltersTotal,"=",config.model.numFilters,"*",len(config.model.filterSizes)
self.hPool.shape
tf.concat(pooledOutputs, 3).shape,tf.concat(pooledOutputs, -1).shape
self.hPoolFlat.shape
self.hDrop.shape
self.logits.shape


# In[94]:


with tf.Session() as sess:
    fd={self.inputX:inpX,self.inputY:inpY,self.dropoutKeepProb:0.8}
    sess.run(tf.global_variables_initializer())
    sess.run(self.embeddedWordsExpanded,feed_dict=fd).shape
    filterShape
    sess.run(W).shape
    sess.run(convRes,feed_dict=fd).shape
    sess.run(pooled,feed_dict=fd).shape
    sess.run(outputW,fd).shape
    sess.run(outputB,fd).shape
    sess.run(self.logits,fd).shape
    sess.run(self.predictions,fd).shape
    sess.run(self.inputY,fd).shape
    sess.run(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY),fd)
    


# In[16]:


tf.reset_default_graph() 


# In[21]:


trainReviews.shape
config.batchSize


# In[61]:


from zac_pyutils import TFUtils


# In[73]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.reset_default_graph() # 防止运行多次后session创建、运行会变慢
ckpt_fp="/home/zhoutong/nlp/data/textcnn/textcnn_model_ckpt/model.ckpt-20480"
pb_fp = "/home/zhoutong/nlp/data/textcnn/tmp_model.pb.sample"
saver = tf.train.import_meta_graph(ckpt_fp + '.meta', clear_devices=True)
with tf.Session() as sess:
    saver.restore(sess, ckpt_fp)
    output_tensor = sess.graph.get_tensor_by_name('output/predictions:0')
    output_tensor
    output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=['output/predictions'])
    with tf.gfile.GFile(pb_fp, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出


# In[ ]:





# In[ ]:





# In[ ]:


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
TFUtils.convert_ckpt2pb(ckpt_fp="/home/zhoutong/nlp/data/textcnn/textcnn_model_ckpt/model.ckpt-20480",
                        pb_fp = "/home/zhoutong/nlp/data/textcnn/tmp_model.pb.sample",
                        output_name_list=['output/predictions'])


# In[ ]:




