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
from collections import Counter


# In[2]:


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


# # Config

# In[3]:


class TrainingConfig(object):
    epoches = 5
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001
    minWordCnt = 5

class ModelConfig(object):
    embeddingSize = 200
    numFilters = 128

    filterSizes = [2, 3, 4, 5]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    _job = "taste"
    _basePath = "/home/zhoutong/nlp/data"
    dataSource = _basePath + "/labeled_{}_train.json".format(_job)
    dataSource = dataSource + ".sample_h10k"
    testSource = _basePath + "/labeled_{}_test.json".format(_job)

    weDim = 300
    weFilePath = _basePath + "/wordEmbeddingInfo"  # \t分割 词,idx,embedding
    ft_modelPath = _basePath + '/cc.en.300.bin'

    batchSize = 128
    pad_size = 1024
    pad = '<PAD>'
    pad_initV = np.zeros(weDim)
    unk = '<UNK>'
    unk_initV = np.random.randn(weDim)


    numClasses = 1  # 二分类设置为1，多分类设置为类别的数目

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
        return res
    @staticmethod
    def pad_list(inp_list,width,pad_const):
        if len(inp_list) >= width:
            return inp_list[:width]
        else:
            return inp_list+[pad_const]*(width-len(inp_list))
    
config = Config()


# # TextCNN

# In[4]:


class TextCNN(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.pad_size], name="inputX")
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
                filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                convRes = tf.nn.conv2d(
                    inpput=self.embeddedWordsExpanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(convRes, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.pad_size - filterSize + 1, 1, 1],
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


# In[5]:


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

# In[11]:


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource

        self.testRatio = config.testRatio
        self._we_fp = config.weFilePath
        self.ft_modelPath = config.ft_modelPath
        self.ft_model = None
        self.totalWordCnt = 0

        self.trainReviews = np.array([])
        self.trainLabels = np.array([])

        self.evalReviews = np.array([])
        self.evalLabels = np.array([])

        self.wordEmbedding = None

        self.labelList = []


    def _readData(self, filePath):
        f_iter = ExqUtils.load_file_as_iter(filePath)
        tokens_list = deque()
        label_list = deque()
        zprint("loading data from: "+filePath)
        for l in tqdm(f_iter):
            text,label = l.strip().split("__label__")
            tokens = Utils.pad_list(Utils.clean_punctuation(text).split(" "),width=self.config.pad_size,pad_const=self.config.pad)
            tokens_list.append(tokens)
            label_list.append(label)
        return np.array(tokens_list), np.array(label_list).reshape(-1,1)

    def _initStopWord(self, stopWordPath):
        with open(stopWordPath, "r") as fr:
            self._stopWordSet = set(fr.read().splitlines())

    def _initFasttextModel(self):
        if self.ft_model is None:
            zprint("init fasttext model")
            self.ft_model = fasttext.load_model(self.ft_modelPath)

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        self._initFasttextModel()

        # 初始化数据集
        tokens_arr, label_arr = self._readData(self._dataSource)

        # tokens 到 emb 存在文件里
        zprint("set所有词")
        tokensSet = set(t for tokens in tokens_arr for t in tokens)
        zprint("预测词向量存入文件: {}".format(self._we_fp))
        with open(self._we_fp,"w") as fw:
            # 加上 <PAD> 和 <UNK> 及其初始化
            for idx,token in tqdm(enumerate(tokensSet),total=len(tokensSet)):
                if token == self.config.pad:
                    emb = self.config.pad_initV
                elif token == self.config.unk:
                    emb = self.config.unk_initV
                else:
                    emb = self.ft_model[token]
                fw.writelines(str(idx)+"\t"+token+"\t"+",".join([str(i) for i in list(emb)])+"\n")


        self.totalWordCnt = len(tokensSet)


        # 初始化训练集和测试集
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.testRatio, random_state=2019)
        train_idx, test_idx = list(sss.split(np.zeros(label_arr.shape[0]), label_arr))[0]
        
        self.trainReviews = tokens_arr[train_idx]
        self.trainLabels = label_arr[train_idx]

        self.evalReviews = tokens_arr[test_idx]
        self.evalLabels = label_arr[test_idx]


# # Produce Data

# In[12]:


data = Dataset(config)
data.dataGen()
print("train data shape: {}".format(data.trainReviews.shape))
print("train label shape: {}".format(data.trainLabels.shape))
print("eval data shape: {}".format(data.evalReviews.shape))
print("wordEmbedding info file: {}".format(data._we_fp))


# In[15]:


tokens_arr, label_arr = data._readData(data._dataSource)


# In[18]:


tokens_arr
set(np.unique(tokens_arr))
set(np.unique(trainLabels))


# In[8]:


trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels


# In[14]:


trainReviews
trainLabels


# # 开始构建计算图

# In[31]:


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


# In[ ]:


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        cnn = TextCNN(config, wordEmbedding)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", cnn.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        savedModelPath = "../model/textCNN/savedModel"
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()

            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)


            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                              labels=labelList)

            trainSummaryWriter.add_summary(summary, step)

            return loss, acc, prec, recall, f_beta


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions],
                feed_dict)

            if config.numClasses == 1:

                acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
            elif config.numClasses > 1:
                acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, precision, recall, f_beta


        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(
                                                                                                             precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))

                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/textCNN/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(cnn.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()
    


# In[ ]:




