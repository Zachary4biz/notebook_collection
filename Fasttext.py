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


# In[73]:


import fasttext
import json
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
import numpy as np
import itertools
import re
from tqdm.auto import tqdm
from zac_pyutils import ExqUtils  # from pip install
from zac_pyutils.ExqUtils import zprint
import os
from collections import deque
import multiprocessing
import xgboost as xgb
import pickle
from sklearn.preprocessing import OneHotEncoder
from gensim.models.wrappers import FastText


# In[113]:


def clean_punctuation(inp_text):
    res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./“”]", r' ', inp_text)
    res = re.sub(r"\\u200[Bb]", r' ', res)
    res = re.sub(r"\n+", r" ", res)
    res = re.sub(r"\s+", " ", res)
    return res


# In[114]:


import time
import random
item = random.shuffle([(1,2),(1,3),(1,4),(2,2),(2,3),(2,4)])


filePath = "/home/zhoutong/nlp/data/labeled_taste_test.json_down_sampled"
f_iter = ExqUtils.load_file_as_iter(filePath)
tokens_list = deque()
label_list = deque()
for l in tqdm(list(f_iter)[:500]):
    text,label = l.split("__label__")
    tokens = clean_punctuation(text).split(" ")
    tokens_list.append(tokens)
    label_list.append(label)
tokens_arr, label_arr = np.array(tokens_list), np.array(label_list)


# In[115]:


enc = OneHotEncoder()

enc.fit(label_arr.reshape([-1,1]))
enc.n_values_
label_onehot_arr = enc.transform(label_arr.reshape(-1,1)).toarray()
label_onehot_arr


# In[80]:


model = fasttext.load_model("/home/zhoutong/nlp/data/cc.en.300.bin")


# In[93]:


tokens = tokens_arr[0]
len(tokens)
len([model[t] for t in tokens])
np.mean([model[t] for t in tokens], axis=0)


# In[99]:


np.mean([np.array([1,2,3]),np.array([3,2,7])], axis=0)
np.concatenate([np.array([1,2,3]),np.array([3,2,7])], axis=0)
np.stack([np.array([1,2,3]),np.array([3,2,7])], axis=0)
np.array([np.array([1,2,3]),np.array([3,2,7])])


# In[122]:


tokensSet = set(t for tokens in tokens_arr for t in tokens)
list(zip([1,2,3],range(3)))
tokens2idx = list(zip(tokensSet,range(len(tokensSet))))


# In[123]:


tokens2idx


# In[70]:


label_onehot_arr.shape[0]


# In[71]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2019)
train_idx, test_idx = list(sss.split(np.zeros(label_onehot_arr.shape[0]), label_onehot_arr))[0]


# In[72]:


list(label_arr[test_idx]) == [label_list[i] for i in test_idx]
label_arr[test_idx]


# In[20]:


allWords = [token for tokens in tokens_list for token in tokens]
subWords = [word for word in allWords if word not in {}]

wordCount = Counter(subWords)

words = [w for w,cnt in wordCount.items() if cnt >= 5]
words


# In[ ]:


filePath = "/home/zhoutong/nlp/data/labeled_taste_test.json_down_sampled"


# In[10]:


base_p = "/home/zhoutong/nlp/data" # "/Users/zac/Downloads/data" /home/zhoutong/nlp/data /data/work/data
p_origin = base_p + "/labeled_timeliness_region_taste_emotion_sample.json"
f_iter = ExqUtils.load_file_as_iter(p_origin)
job = "taste"
job_idx_file = os.path.join(base_p,"sampleIdx_{}".format(job))

cnt = 0
for idx,line in tqdm(enumerate(f_iter)):
    info = json.loads(line)
    print(str(idx)+"\t"+str(info[job]))
    cnt += 1
    if cnt>10:
        break


# In[3]:


class EmbModel(object):
    def __init__(self):
        self.model_param = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 20,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'nthread': None,
        }
        self.params_grid = {
            'max_depth': [1, 2, 3, 4, 5, 6],
            'n_estimators': [10, 15, 20, 50, 52, 55, 60, 70, 80],
        }
        self.model_emb = None
        self._clf = None

    def load_word_embedding(self, model_path):
        zprint("loading word_embedding model. from: {}".format(model_path))
        self.model_emb = fasttext.load_model(model_path)

    # 清理符号
    @staticmethod
    def _clean_text(inp_text):
        res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./]", r' ', inp_text)
        res = re.sub(r"\n+", r" ", res)
        res = re.sub(r"\s+", " ", res)
        res = res.strip()
        return res

    def _get_article_vector(self, text):
        if self.model_emb is None:
            raise Exception("self.model_emb is None. use 'MyModel.load_word_embedding()' to load embedding model")
        else:
            return np.mean([self.model_emb.get_sentence_vector(sen) for sen in text.split(".")], axis=0)


    def fit(self, text_list, label_list, weight_list):
        input_vec_list = np.array([self._get_article_vector(self._clean_text(text)) for text in text_list])
        self._clf = GridSearchCV(xgb.sklearn.XGBClassifier(**self.model_param), self.params_grid, verbose=1, cv=4, scoring='roc_auc')
        self._clf.fit(input_vec_list, np.array(label_list), sample_weight=np.array(weight_list))

    def train_supervised(self, fasttext_format_sample_path,weight_dict=None,label_prefix="__label__"):
        with open(fasttext_format_sample_path, "r") as f:
            content = [i.strip().split(label_prefix) for i in f.readlines()]
        text_list = [text for text,_ in content]
        label_list = [label_prefix+label_ for _,label_ in content]
        weight_list = [1.0 for _ in label_list]
        if weight_dict is not None:
            weight_list = [weight_dict[label] for label in label_list]
        self.fit(text_list, label_list, weight_list)
        return self

    def predict(self, text):
        input_vec = self._get_article_vector(self._clean_text(text))
        if self._clf is None:
            raise Exception("self._clf is None. use 'MyModel.fit()' or 'MyModel.load()' to init self._clf")
        return self._clf.predict(input_vec)

    def save(self, save_path):
        pickle.dump(self._clf, open(save_path, "wb"))

    def load(self, load_path):
        self._clf = pickle.load(open(load_path, "rb"))
        return self


# In[3]:


base_p = "/home/zhoutong/nlp/data" # /Users/zac/Downloads/data  /home/zhoutong/nlp/data  /data/work/data
job = "taste" # timeliness taste emotion region(1,0)
# 正式数据
p = base_p+"/labeled_timeliness_region_taste_emotion_sample.json"
# 准备词向量训练样本
prepare_samples_corpus = False
p_train_corpus = base_p + "/corpus4we.text"
# 直接按不均衡样本训练
prepare_samples = True # 
p_train = base_p+"/labeled_{}_train.json".format(job)
p_test = base_p+"/labeled_{}_test.json".format(job)
model_path = base_p+"/{}_model.ftz".format(job)
# 亚采样
prepare_samples_downsamples = True
p_train_downsample = p_train + "_down_sampled"
p_test_downsample = p_test + "_down_sampled"
model_path_downsample = base_p+"/{}_model_down_sampled.ftz".format(job)
downsample_queue_dict = {
    'timeliness':dict((str(i),deque([], 2900)) for i in range(1,9)),
    'taste':dict((str(i),deque([], 61508)) for i in range(0,4)),
    'emotion':dict((str(i),deque([], 43642)) for i in range(0,3)),
    'region':dict((str(i),deque([], 220591)) for i in range(0,2)),
}
# 过采样
prepare_samples_oversamples = True
p_train_oversample = p_train + "_oversample"
p_test_oversample = p_test + "_oversample"
model_path_oversample = base_p+"/{}_model_oversample.ftz".format(job)
oversample_queue_dict = {
    'timeliness':dict((str(i),deque([], 40*10000)) for i in range(1,9)),
    'taste':dict((str(i),deque([], 30*10000)) for i in range(0,4)),
    'emotion':dict((str(i),deque([], 40*10000)) for i in range(0,3)),
    'region':dict((str(i),deque([], 40*10000)) for i in range(0,2)),
}
# 使用 wordEmbedding & XGB 做分类
use_EmbModel = True
we_model_path = base_p + "/cc.en.300.bin"
model_path_we = base_p+"/{}_model_we.ftz".format(job)
total_weight_dict = {
    'timeliness':{'__label__1':11.0,'__label__2':1.0,'__label__3':7.0,'__label__4':25.0,'__label__5':97.0,'__label__6':153.0,'__label__7':18.0,'__label__8':9.0},
    'taste':{'__label__0':3.25,'__label__1':1.0,'__label__2':4.0,'__label__3':6.25},
    'emotion':{'__label__0':4.0,'__label__1':11.0,'__label__2':1.0},
    'region':{'__label__0':2.0,'__label__1':1.0}
}


# In[4]:


class Util():
    # 清理符号
    @staticmethod
    def clean_text(inp_text):
        res = re.sub(r"[~!@#$%^&*()_+-={\}|\[\]:\";'<>?,./]", r' ', inp_text)
        res = re.sub(r"\n+", r" ", res)
        res = re.sub(r"\s+", " ", res)
        return res
    # fasttext自带的测试API
    @staticmethod
    def fasttext_test(model, file_p):
        n, precision, recall = model.test(file_p)
        zprint("test 结果如下:")
        zprint('P@1:'+str(precision))  # P@1 取排第一的分类，其准确率
        zprint('R@1:'+str(recall))  # R@1 取排第一的分类，其召回率
        zprint('Number of examples: {}'.format(n))
        zprint(model.predict("I come from china"))
    # 自定义验证各类别的 recall percision f1
    @staticmethod
    def metric_on_file(label_pred_list):
        all_label = set(i[0] for i in label_pred_list)
        res = []
        for curLbl in sorted(all_label):
            TP = sum(label == pred == curLbl for label, pred in label_pred_list)
            label_as_curLbl = sum(label == curLbl for label, pred in label_pred_list)
            pred_as_curLbl = sum(pred == curLbl for label, pred in label_pred_list)
            P = TP / pred_as_curLbl if TP > 0 else 0.0
            R = TP / label_as_curLbl if TP > 0 else 0.0
            F1 = 2.0 * P * R / (P + R) if TP > 0 else 0.0
            res.append((curLbl,R,P,F1))
        res.append(('__label__M', sum(R for _,R,_,_ in res)/len(res) ,sum(P for _,_,P,_ in res)/len(res), sum(F1 for _,_,_,F1 in res)/len(res)))
        for curLbl,R,P,F1 in res:
            print("[label]: {}, [recall]: {:.4f}, [precision]: {:.4f}, [f1]: {:.4f}".format(curLbl,R,P,F1))

        label_grouped = itertools.groupby(sorted([label for label, pred in label_pred_list]))
        pred_grouped = itertools.groupby(sorted([pred for label, pred in label_pred_list]))
        label_distribution = dict((k, len(list(g))) for k, g in label_grouped)
        pred_distribution = dict((k, len(list(g))) for k, g in pred_grouped)
        print("[label分布]: ", label_distribution)
        print("[pred分布]: ", pred_distribution)


# In[7]:


##############################################################################################################
# 分析样本分布
# elapsed: roughly 29.4s
# timeliness {'1': 39566, '2': 456327, '3': 64505, '4': 17625, '5': 4698, '6': 2979, '7': 24271, '8': 49380}
# emotion {'0': 122369, '1': 43642, '2': 493340}
# taste {'0': 117872, '1': 384200, '2': 95771, '3': 61508}
# region {'0': 438760, '1': 220591}
###############################################################################################################
find_distribution = False
all_job = ['timeliness','emotion','taste','region']
if find_distribution:
    content_iter = ExqUtils.load_file_as_iter(p)
    ori_distribution = {'timeliness': {}, 'emotion': {}, 'region': {}, 'taste': {}}
    while True:
        data = list(itertools.islice(content_iter, 10000 * 10))
        if len(data) > 0:
            json_res = [json.loads(i.strip()) for i in data]
            # sample_list = [c['title'] + ". " + c['text'] for c in content]
            for job in all_job:
                job_label_list = np.asarray(sorted([str(c[job]) for c in json_res]))
                for k, g in itertools.groupby(job_label_list):
                    ori_distribution[job].update({k: len(list(g)) + ori_distribution[job].get(k, 0)})
        else:
            break
    for job in all_job:
        print(job,ori_distribution[job])


# In[ ]:


####################
# 准备 词向量 训练样本
####################
if prepare_samples_corpus:
    print("提取文本语料用于训练词向量")
    print("清空文件")
    print(os.popen('> ' + p_train_corpus), p_train_corpus)
    with open(p,"r") as f:
        json_res = [json.loads(i.strip()) for i in f.readlines()]
    text_list = [c['title'] + ". " + c['text'] for c in json_res]
    text_list = [clean_text(i)+"\n" for i in text_list]
    with open(p_train_corpus,"w") as f:
        f.writelines(text_list)

####################
# 准备（分类）训练样本
# {'1': 39566, '2': 456327, '3': 64505, '4': 17625, '5': 4698, '6': 2979, '7': 24271, '8': 49380}
####################
if prepare_samples:
    print("加载各样本")
    content_iter = ExqUtils.load_file_as_iter(p)
    distribution = {}
    print("清空文件")
    print(os.popen('> '+p_train),p_train)
    print(os.popen('> '+p_test),p_test)
    while True:
        data = list(itertools.islice(content_iter, 10000 * 15))
        if len(data) > 0:
            json_res = [json.loads(i.strip()) for i in data]
            sample_list = [c['title'] + ". " + c['text'] for c in json_res]
            job_label_list = np.asarray(sorted([str(c[job]) for c in json_res]))
            for k, g in itertools.groupby(job_label_list):
                distribution.update({k: len(list(g)) + distribution.get(k, 0)})
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            train_idx, test_idx = list(sss.split(sample_list, job_label_list))[0]
            with open(p_train, "a") as f:
                for idx in tqdm(train_idx):
                    to_write = clean_text(sample_list[idx]) + "__label__" + job_label_list[idx]
                    f.writelines(to_write + "\n")
            with open(p_test, "a") as f:
                for idx in tqdm(test_idx):
                    to_write = clean_text(sample_list[idx]) + "__label__" + job_label_list[idx]
                    f.writelines(to_write + "\n")
        else:
            break
    total = sum(list(distribution.values()))
    print(">>> 整体（训练+测试）样本分布："+str([(k, round(v / total, 4)) for k, v in distribution.items()]))

###############################################
# 准备（分类）训练样本
# 对不均衡的数据: 亚采样多数类
###############################################
if prepare_samples_downsamples:
    print("downsample, 加载各样本")
    content_iter = ExqUtils.load_file_as_iter(p)
    samples_dict = downsample_queue_dict[job]
    print("清空文件")
    print(os.popen('> '+ p_train_downsample), p_train_downsample)
    print(os.popen('> '+ p_test_downsample), p_test_downsample)
    # 加载文件遍历进行FIFO
    while True:
        data = list(itertools.islice(content_iter, 5000 * 1))
        if len(data) > 0:
            content = [json.loads(i.strip()) for i in data]
            for c in content:
                text = c['title']+" "+c['text']
                label = str(c[job])
                samples_dict[label].append(text)
        else:
            break
    # 数据拆分: {label: list(text)} -> text_list & label_list
    text_list, label_list = deque(), deque()
    for k, v in samples_dict.items():
        text_list.extend(v)
        label_list.extend([k] * len(v))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, test_idx = list(sss.split(text_list, label_list))[0]
    with open(p_train_downsample, "a") as f:
        for idx in tqdm(train_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")
    with open(p_test_downsample, "a") as f:
        for idx in tqdm(test_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")
            
###############################################
# 准备（分类）训练样本
# 对不均衡的数据: 过采样少数类
###############################################
if prepare_samples_oversamples:
    print("oversample 加载各样本")
    content_iter = ExqUtils.load_file_as_iter(p)
    samples_dict = oversample_queue_dict[job]
    print("清空文件")
    print(os.popen('> ' + p_train_oversample), p_train_oversample)
    print(os.popen('> ' + p_test_oversample), p_test_oversample)
    # 加载文件遍历进行FIFO
    while True:
        data = list(itertools.islice(content_iter, 100000 * 2))
        if len(data) > 0:
            content = [json.loads(i.strip()) for i in data]
            for c in content:
                text = c['title'] + " " + c['text']
                label = str(c[job])
                samples_dict[label].append(text)
        else:
            # 循环结束时，扩充为填满的类别（oversampling）
            for label, text_deque in samples_dict.items():
                while len(text_deque)<text_deque.maxlen:
                    text_deque.extend(text_deque)
                print("    {} oversampling完毕, 当前deque长度: {}".format(label,len(text_deque)))
            break
    text_list, label_list = deque(), deque()
    for k, v in samples_dict.items():
        text_list.extend(v)
        label_list.extend([k] * len(v))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, test_idx = list(sss.split(text_list, label_list))[0]
    with open(p_train_oversample, "a") as f:
        for idx in tqdm(train_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")
    with open(p_test_oversample, "a") as f:
        for idx in tqdm(test_idx):
            to_write = clean_text(text_list[idx]) + "__label__" + label_list[idx]
            f.writelines(to_write + "\n")


# In[6]:


####################
# 训练集、模型参数配置
####################
train_path = p_train # p_train_oversample p_train_downsample p_train
test_path = p_test # p_test_oversample p_test_downsample p_test
persist_path = model_path_we # model_path_oversample model_path_downsample model_path_we model_path
print("[train_path]: {}\n[test_path]: {}\n[model_path]: {}".format(train_path,test_path,persist_path))


# In[7]:


label_prefix = "__label__"
weight_dict = total_weight_dict[job]

with open(train_path, "r") as f:
    content = [i.strip().split(label_prefix) for i in f.readlines()]
text_list = [text for text,_ in content]
label_list = [label_prefix+label_ for _,label_ in content]
weight_list = [1.0 for _ in label_list]
if weight_dict is not None:
    weight_list = [weight_dict[label] for label in label_list]


# In[8]:


text_list[0]


# In[9]:


label_list[0]


# In[10]:


print("a")


# In[ ]:





# In[ ]:





# In[ ]:


if use_EmbModel :
    print("use_EmbModel")
    model = EmbModel()
    model.load_word_embedding(we_model_path)
    print("use weight as: " + str(total_weight_dict[job]))
    model.train_supervised(fasttext_format_sample_path=train_path,weight_dict=total_weight_dict[job])
    model.save(persist_path)
else:
    #######################
    # 有监督（分类）模型训练
    #######################
    zprint("开始训练有监督（分类）模型...")
    supervised_params = {
        # 'input': '',
        'lr': 0.01,  # 学习率
        'dim': 180,  # 词向量维数
        'ws': 5,  # 上下文窗口
        'epoch': 15,  # epoch
        'minCount': 10,  # 每个词最小出现次数
        'minCountLabel': 0,  # 每个label最小出现次数
        'minn': 2,  # 字符级别ngram的最小长度
        'maxn': 4,  # 字符级别ngram的最大长度
        'neg': 5,  # 负采样个数
        'wordNgrams': 3,  # 词级别ngram的个数
        'loss': 'softmax',  # 损失函数 {ns, hs, softmax, ova}
        'bucket': 2000000,  # buckets个数， 所有n-gram词hash到bucket里
        'thread': 8,  # 线程
        'lrUpdateRate': 100,  # change the rate of updates for the learning rate [100]
        't': 0.0001,  # sampling threshold [0.0001]
        'label': '__label__',  # label prefix ['__label__']
        'verbose': 2,  # verbose [2]
        'pretrainedVectors': ''  # pretrained word vectors (.vec file) for supervised learning []
    }
    clf = fasttext.train_supervised(input=train_path, **supervised_params)
    zprint("总计产生词条：{}个，标签： {}个".format(len(clf.words), len(clf.labels)))
    zprint("各个标签为：{}".format(", ".join(clf.labels)))


    ##############
    # 分类模型测试
    ##############
    test_on_model(clf,test_path)

    #################
    # 压缩 & 保存 模型
    #################
    quantization = True
    if quantization:
        zprint("压缩模型")
        clf.quantize(train_path, retrain=True)
    zprint("保存模型..")
    clf.save_model(persist_path)


# In[ ]:


##############
# 分类模型测试
##############
test_on_model(clf,test_path)


# In[ ]:


#################
# 压缩 & 保存 模型
#################
quantization = True
if quantization:
    zprint("压缩模型")
    clf.quantize(train_path, retrain=True)
zprint("保存模型..")
clf.save_model(persist_path)


# In[ ]:


#################
# 分类模型测试 自测
#################
model = fasttext.load_model(persist_path)
sep = '__label__'
with open(test_path, "r") as f:
    content = [i.strip() for i in f.readlines()]

label_pred_list = []
for i in tqdm(content):
    text = clean_text(i.strip().split(sep)[0])
    label = sep + i.strip().split(sep)[1]
    y_pred = model.predict(text)[0][0]
    label_pred_list.append((label,y_pred))

all_label = set(i[0] for i in label_pred_list)
for curLbl in all_label:
    TP = sum(label == pred == curLbl for label,pred in label_pred_list)
    label_as_curLbl = sum(label == curLbl for label,pred in label_pred_list)
    pred_as_curLbl = sum(pred == curLbl for label,pred in label_pred_list)
    P = TP / pred_as_curLbl if TP>0 else 0.0
    R = TP / label_as_curLbl if TP>0 else 0.0
    F1 = 2.0*P*R/(P+R) if TP>0 else 0.0
    print("[label]: {}, [recall]: {:.4f}, [precision]: {:.4f}, [f1]: {:.4f}".format(curLbl,R,P,F1))
    
label_grouped = itertools.groupby(sorted([label for label,pred in label_pred_list]))
pred_grouped = itertools.groupby(sorted([pred for label,pred in label_pred_list]))
label_distribution = dict((k,len(list(g))) for k,g in label_grouped)
pred_distribution = dict((k,len(list(g))) for k,g in pred_grouped)
print("[label分布]: ", label_distribution)
print("[pred分布]: ", pred_distribution)


# In[5]:


premodel = fasttext.load_model("/home/zhoutong/nlp/data/taste_model_oversample.ftz")


# In[7]:


sep = "__label__"
with open("/home/zhoutong/nlp/data/labeled_taste_train.json_oversample", "r") as f:
    content = [i.strip() for i in f.readlines()]


# In[9]:


p_train
model_path_oversample


# In[ ]:


for i in tqdm(content):
    text = Util.clean_text(i.strip().split(sep)[0])
    label = sep + i.strip().split(sep)[1]
    y_pred = premodel.predict(text)[0][0]

