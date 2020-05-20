#!/usr/bin/env python
# coding: utf-8

# In[73]:


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


# In[74]:


import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt


# In[16]:


baseDir="/home/zhoutong/notebook_collection/tmp/NLP_ner/lstmcrf"
checkpoint_dir = baseDir+"/ckpt"


# # SubclassedM

# ## BiLSTMCRF

# In[380]:


# 参考： https://github.com/saiwaiyanyu/bi-lstm-crf-ner-tf2.0/blob/master/train.py
class LSTMCRF(tf.keras.Model):
    def __init__(self, label_size, lstm_units, vocab_size, emb_dim):
        super().__init__()
        self.label_size = label_size
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.emb_dim)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True))
#         self.dense = tf.keras.layers.Dense(label_size,activation="softmax")
        self.dense = tf.keras.layers.Dense(label_size) # 正统做法是没有激活函数没做归一化
        self.dropout = tf.keras.layers.Dropout(0.5)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(self.label_size, self.label_size)),trainable=False)

    # @tf.function
    def call(self, text,labels=None,training=None):
        seq_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        # -1 change 0·
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        logits = self.dense(self.bilstm(inputs))

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits, label_sequences, seq_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, seq_lens, log_likelihood
        else:
            return logits, seq_lens
        


# ## BERTLayer

# In[20]:


import tensorflow_hub as hub


# In[ ]:


max_seq_length = 128  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])


# In[ ]:


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables
        
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        
        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
        
        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        
        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
        
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# In[ ]:





# # LoadData

# In[489]:


pad_length = 100
target_fp=baseDir+"/data/renminribao/target_BIO_2014_cropus.txt"
source_fp=baseDir+"/data/renminribao/source_BIO_2014_cropus.txt"
with open(target_fp,"r") as fr:
    target=[i.strip() for i in fr.readlines()]
with open(source_fp,"r") as fr:
    source=[i.strip() for i in fr.readlines()]

# 所有样本
sample_iter=zip(source,target)
# 词表
all_words=set([char for sentence in source for char in sentence.split(" ") if char != ''])
word2idx = dict((word,idx+1) for idx,word in enumerate(all_words))
word2idx.update({"PAD":0})
# 标注label
all_tags=set([tag for tags in target for tag in tags.split(" ") if tag != ''])
tag2idx = dict((tag,idx) for idx,tag in enumerate(all_tags))
# tag2idx.update({"PAD":0})

# 训练集测试集
def _yield_samples(source_inp,target_inp):
    for sen,labels in zip(source_inp,target_inp):
        X = ([w for w in sen.split(" ")]+['PAD']*pad_length)[:pad_length]
        X = [word2idx[w] for w in X]
        Y = ([w for w in labels.split(" ")]+['O']*pad_length)[:pad_length]
        Y = [tag2idx[l] for l in Y]
        yield (X,Y)

total_size = len(source)
train_size = int(total_size*0.7)
train_dataset = _yield_samples(source[:train_size],target[:train_size])
test_dataset = _yield_samples(source[train_size:],target[train_size:])


# In[482]:


idx2word=dict([(v,k) for k,v in word2idx.items()])
idx2tag=dict([(v,k) for k,v in tag2idx.items()])


# In[475]:


sample = sample_iter.__next__()
sample
sen,tag = sample
sen_idx=[word2idx[i] for i in sen.split(" ")]
tag_idx=[tag2idx[i] for i in tag.split(" ")]
print(sen,"\n",sen_idx)
print(tag,"\n",tag_idx)

sen_idx = sen_idx[:4]
tag_idx = tag_idx[:4]
sen_idx,tag_idx


# # InitModel | Params

# In[511]:


config={
    "label_size":len(tag2idx),
    "lstm_units":8,
    "vocab_size":len(word2idx),
    "emb_dim":300
}
print(config)
M = LSTMCRF(**config)

M.build((None,20))
M.summary()


# # Train

# ## pre

# In[525]:


##################################################
# tf.train.Checkpoint 不能存模型内部有Model类变量的模型
# tf.train.CheckpointManager 不能在保存时指定prefix 
# 自行包装一个类，主要使用model.save_weights
##################################################
class CustomCkpt:
    def __init__(self, ckpt_dir, model=None,max_keep=20):
        self.ckpt_dir = ckpt_dir
        self.best_valid_acc = 0.0
        self.best_valid_loss = 1e10
        self.total_saved_fps = []
        self.history = []
        self.model = model
        self.max_keep=max_keep
         
    def save(self,val_acc,val_loss,model=None,fileName=None):
        if model is not None:
            self.model = model
        assert self.model is not None, "初始化时未指定model则.save()必须提供model"
        if fileName is None:
            save_fp = os.path.join(self.ckpt_dir,"ckpt_"+len(self.total_saved_fps))
        else:
            save_fp = os.path.join(self.ckpt_dir,fileName)
        saved = False # 因为acc和loss提升时都要打log并更新best，但是保存只存一次
        if val_acc > self.best_valid_acc:
            print(f"acc improved [from]:{self.best_valid_acc:.4f} [to]:{val_acc:.4f}.")
            self.best_valid_acc = val_acc
            if not saved:
                self.model.save_weights(save_fp)
                saved=True
        if val_loss < self.best_valid_loss:
            print(f"loss improved [from]:{self.best_valid_loss:.4f} [to]:{val_loss:.4f}.")
            self.best_valid_loss = val_loss
            if not saved:
                self.model.save_weights(save_fp)
                saved=True
        # 限制最多保存文件个数
        if saved:
            print(f"[ckpt-path]: {save_fp}")
            self.total_saved_fps.append(save_fp)
            if len(self.total_saved_fps) >= self.max_keep:
                toDel_fp = self.total_saved_fps.pop(0)
                status,output=subprocess.getstatusoutput(f"rm {toDel_fp}*")
        
        self.history.append({"val_acc":val_acc, "val_loss":val_loss})
        


# In[526]:


######
# tbd
######
opt=tf.keras.optimizers.Adam()
tbd_dir = baseDir+"/tensorboard"
if os.path.exists(tbd_dir):
    import shutil
    shutil.rmtree(tbd_dir)
    print("历史tbd信息已删除")
summary_writer = tf.summary.create_file_writer(tbd_dir)
with summary_writer.as_default():
    pass
#     _=tf.summary.image("Trainning Data", normal_flow_train[0][0], max_outputs=4, step=0)

    
#######
# ckpt
#######
ckpt_saver = CustomCkpt(checkpoint_dir)

##################
# 损失函数和评价指标
##################
ce_loss_fn = tf.keras.losses.categorical_crossentropy
acc_fn = tf.keras.metrics.categorical_accuracy

########
# 优化器
########
optimizer = tf.optimizers.Adam(1e-3)


# ## train loop

# In[527]:


data_batch = list(itertools.islice(train_dataset,batch_size))
text_batch = np.array([i[0] for i in data_batch])
label_batch = np.array([i[1] for i in data_batch])
text_batch
text_batch.shape


# In[528]:


def train_one_step(text_batch, labels_batch):
    with tf.GradientTape() as tape:
        logits, seq_len_batch, log_likelihood = M(text_batch, labels_batch,training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, M.trainable_variables)
    optimizer.apply_gradients(zip(gradients, M.trainable_variables))
    return loss,logits, seq_len_batch

def get_acc_one_step(logits, seq_len_batch, labels_batch):
    paths = []
    acc = 0
    for logit, seq_len, labels in zip(logits, seq_len_batch, labels_batch):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:seq_len], M.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:seq_len]], padding='post'),
                                 dtype=tf.int32)
        )
        acc = acc + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = acc / len(paths)
    return acc


def get_validation_acc():
    while True:
        data_batch = list(itertools.islice(test_dataset,batch_size))
        if len(data_batch) < batch_size:
            break
        text_batch = np.array([i[0] for i in data_batch])
        labels_batch = np.array([i[1] for i in data_batch])
        step = step + 1
        acc = get_acc_one_step(logits, seq_len_batch, labels_batch)
        


best_acc = 0
step = 0
epoch=10
batch_size=20
for epoch in range(epoch):
    while True:
        data_batch = list(itertools.islice(train_dataset,batch_size))
        if len(data_batch) < batch_size:
            break
        text_batch = np.array([i[0] for i in data_batch])
        labels_batch = np.array([i[1] for i in data_batch])
        step = step + 1
        loss, logits, seq_len_batch = train_one_step(text_batch, labels_batch)
        if step % 20 == 0:
            acc = get_acc_one_step(logits, seq_len_batch, labels_batch)
            tqdm.write(f"[e]: {epoch} [step]: {step} [acc]: {acc:.4f} [loss]:{loss:.4f}")
            ckpt_saver.save(val_loss=loss,val_acc=acc,model=M,fileName=f"ckpt_e{epoch}_trainLoss{loss:.4f}_trainAcc{acc:.4f}")


# In[ ]:





# # Valid

# In[ ]:





# In[ ]:





# # Test

# In[ ]:





# # CRF 计算参数实验
# 
# ```
# loglikelihood = sequence_score - log_norm
# = (unary_score + binary_score) - logsumexp(alphas)
# = (unary_score + binary_score) - logsumexp(crf_forward)
# ```

# ## loglikelihood

# In[466]:


sen_idx=[500,501,502,503]
# tag_idx=[1,2,3,10]
tag_idx=[1,2,3,9]
inp = np.array([sen_idx])
"inp"
inp
# logits=M.dense(M.dropout(M.bilstm(M.embedding(inp))))
logits,seq_lens=M(inp)
"logits",logits.numpy().shape
logits.numpy()
label_sequences = np.array([tag_idx])
"label_sequences",label_sequences.shape
label_sequences
text_lens=np.array([len(sen_idx)])
loglikelihood,trans_params = tfa.text.crf_log_likelihood(logits, label_sequences, text_lens,M.transition_params)
"loglikelihood"
loglikelihood.numpy()
"seq_score"
tfa.text.crf_sequence_score(logits,label_sequences,text_lens,M.transition_params).numpy()
"log_norm"
tfa.text.crf_log_norm(logits, text_lens, M.transition_params).numpy()


# ## sequence_socre

# In[252]:


f"crf_sequence_score: {tfa.text.crf_sequence_score(logits,label_sequences,text_lens,M.transition_params)}"
"== unary_score+bianry_score"
f"unary_score: {tfa.text.crf_unary_score(label_sequences, text_lens, logits)}"
f"== sum(emission of labelSeq): {sum([logits.numpy()[0,idx,i] for idx,i in enumerate(label_sequences[0])])}"
f"bianry_score: {tfa.text.crf_binary_score(label_sequences, text_lens, M.transition_params)}"
f"== sum(transition of labelSeq): {sum([M.transition_params.numpy()[a,b] for a,b in list(zip(label_sequences[0][:-1],label_sequences[0][1:]))])}"


# ## log_norm

# In[271]:


f"log_norm: {tfa.text.crf_log_norm(logits, text_lens, M.transition_params).numpy()}"


# ### first_input&rest_input

# In[297]:


inputs=logits
sequence_lengths=text_lens
first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
first_input = tf.squeeze(first_input, [1])
rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
f"[logits]: {logits.numpy().shape}"
logits.numpy()
f"[first_input]: {first_input.numpy().shape}"
first_input.numpy()
f"[rest_of_input]: {rest_of_input.numpy().shape}"
rest_of_input.numpy()


# ### crf_forward
# 前向计算和viterbi类似，只不过是把求最大改成了求和

# In[306]:


alphas = tfa.text.crf_forward(rest_of_input, first_input, M.transition_params, sequence_lengths)
alphas
tf.reduce_logsumexp(alphas,[1])


# In[305]:


# 模拟计算所有序列情况各自的sequence_score，并求和
# 从结果来看不是这样计算的
length=text_lens[0]
ps=[]
for i in range(length):
    for j in range(length): 
        for m in range(length):
                ps.append([i,j,m])
ps
res=[]
for seq in ps:
    unary_score=sum([logits.numpy()[0][idx,i] for idx,i in enumerate(seq)])
    binary_score = sum([M.transition_params[a][b].numpy() for a,b in list(zip(seq[:-1],seq[1:]))])
    res.append((unary_score, binary_score))

len(res)
res
# tf.reduce_logsumexp(res).numpy()
# for word_idx,emit_list in enumerate(logits.numpy()[0]):
#     for emit_idx,emit in enumerate(emit_list):
#         emit


# In[296]:


rest_of_input


# In[211]:


# crf_forward
inputs = rest_of_input
stat = first_input
transition_params = M.transition_params
"shape: rest_of_input,first_input,transition_params"
[i.numpy().shape for i in [inputs,stat,transition_params]]

inputs = tf.transpose(inputs, [1, 0, 2])
transition_params = tf.expand_dims(transition_params, 0)
"shape(after): rest_of_input,first_input,transition_params"
[i.numpy().shape for i in [inputs,stat,transition_params]]


last_index = tf.maximum(tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)


stat
tf.expand_dims(stat, 2)
transition_params
tf.expand_dims(stat, 2) + transition_params
tf.reduce_logsumexp(transition_scores, [1])

