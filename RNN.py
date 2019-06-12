#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import matplotlib.pyplot as plt


# # 参考
# - [莫烦 RNN on Tensorflow](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/#%E5%AE%9A%E4%B9%89-RNN-%E7%9A%84%E4%B8%BB%E4%BD%93%E7%BB%93%E6%9E%84)
# - [莫烦 RNN on Pytorch](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-RNN-classification/)
# - [Tensorflow 官方教程](https://www.tensorflow.org/tutorials/sequences/recurrent?hl=zh-cn)

# # Morvan | Tensorflow

# In[3]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed


# ## Data Prepare

# In[3]:


# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ## Model

# ### 超参数

# In[4]:


# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128            
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)


# ### RNN建立计算图

# In[5]:


# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# In[7]:


def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    # 使用 basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state

    # 如果使用tf.nn.dynamic_rnn(cell, inputs), 我们要确定 inputs 的格式. tf.nn.dynamic_rnn 中的 time_major 参数会针对不同 inputs 格式有不同的值.
    # 如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
    # 如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results


# ## Fit

# In[8]:


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1


# # Morvan | Pytorch

# In[41]:


import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter # torch 1.14 才会更新这个
import matplotlib.pyplot as plt
torch.manual_seed(1)    # reproducible


# ## Data Prepare

# In[10]:


DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle
root_path = "/home/zhoutong/data"

# Mnist 手写数字
# transform: 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
train_data = torchvision.datasets.MNIST(
    root=root_path,    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)
test_data = torchvision.datasets.MNIST(root=root_path, train=False)
train_data
test_data


# In[18]:


print(train_data.data.size())     # (60000, 28, 28)
print(train_data.targets.size())   # (60000)
_=plt.imshow(train_data.data[0].numpy(), cmap='gray')
_=plt.title('%i' % train_data.targets[0])
plt.show()


# In[25]:


# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
test_x = test_data.data.type(torch.FloatTensor)[:2000]   # shape (2000, 28, 28) 
test_x = test_x/255. # normalize to range(0,1)
test_y = test_data.targets.numpy()[:2000]    # covert to numpy array


# ## Model

# ### 超参数

# In[5]:


# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28      # rnn 时间步数 / 图片高度
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
LR = 0.01           # learning rate


# ### RNN封装

# In[28]:


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.LSTM(         # if use torch.nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
        )

        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


# ## Fit

# In[29]:


rnn = RNN()
print(rnn)


# In[31]:


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = torch.nn.CrossEntropyLoss()                       # the target label is not one-hotted


# In[32]:


# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# In[33]:


# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


# In[37]:


# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print('pred number: ', pred_y)
print('true number: ', test_y[:10])


# # Tensorflow Official

# In[10]:


import collections


# In[15]:


list(collections.OrderedDict.fromkeys([1,2,3,4,44,56,7,87,4,4,5,6,5]).keys())


# ## Prepare

# In[18]:


import reader
raw_data = reader.ptb_raw_data("/home/zhoutong/data/PTB/simple-examples/data")
train_data, valid_data, test_data, _ = raw_data


# In[19]:


# 数据量太大，继续抽小样本
train_data, valid_data, test_data = train_data[:100], valid_data[:20], test_data[:20]
len(train_data)


# In[21]:


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


# ## Config

# In[22]:


BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000 # 词的id数最大到9999
    rnn_mode = BLOCK
    
config = TestConfig() # 训练时初始化用的config
eval_config = TestConfig() # 验证时用的config
eval_config.batch_size = 1
eval_config.num_steps = 1


# In[ ]:


train_input = PTBInput(config=config, data=train_data, name="TrainInput")
train_input.batch_size
train_input.num_steps
train_input.input_data.graph
tf.get_default_graph()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(train_input.input_data)


# In[ ]:


res


# In[63]:


with tf.get_default_graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")

        #         with tf.variable_scope("Model", reuse=None, initializer=initializer):
        #             m = PTBModel(is_training=True, config=config, input_=train_input)
        #         tf.summary.scalar("Training Loss", m.cost)
        #         tf.summary.scalar("Learning Rate", m.lr)

        # 封装的模型PTBModel的初始化参数
        _is_training = True
        _input = train_input
        config = config
        # 数据的参数
        batch_size = _input.batch_size
        num_steps = _input.num_steps
        # 模型结构
        _rnn_params = None
        _cell = None
        # 超参数
        size = config.hidden_size
        vocab_size = config.vocab_size

        # 构建embedding
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, _input.input_data)
        # 若在训练过程中，使用dropout
        if _is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        # 构建rnn计算图
        def make_cell():
            cell = tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell
        # MultiRNNCell | 多个LSTM内部结构前一个的输出是后一个的输入，inp -> lstm1 -> lstm2 -> lstm3 -> out
        cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # state状态 | 一开始MultiRNNCell都做全零初始化 zero_state
        _initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = _initial_state
        # outputs输出
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        
#         output, state = self._build_rnn_graph(inputs, config, is_training)
        
    with tf.name_scope("Valid"):
        pass

    with tf.name_scope("Test"):
        pass


# In[37]:


with tf.Graph().as_default():
    with tf.Session() as sess:
    sess.run(train_input)
    
input_.input_data


# In[ ]:




