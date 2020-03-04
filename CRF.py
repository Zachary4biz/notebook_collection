#!/usr/bin/env python
# coding: utf-8

# In[8]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import Pool
import copy,os,sys,psutil
from collections import Counter


# In[2]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np


# # 模拟数据准备

# In[3]:


# ########### log_likelihood ##############
# 表示不同骰子投掷出不同点数的概率的log
#  - 第一列是无偏骰子，第二列是有偏骰子
# array([[-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -3.21887582],
#        [-1.79175947, -0.22314355]])
# #########################################
probabilities = {
    'fair': np.array([1 / 6] * 6),  # 无偏骰子
    'loaded': np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.8]),  # 有偏骰子
}
log_likelihood = np.hstack([np.log(probabilities['fair']).reshape(-1, 1),
                            np.log(probabilities['loaded']).reshape(-1, 1)])


# In[4]:


# ########## 预设转移概率矩阵 #########################
# 后面会根据这个矩阵构造样本，CRF可以认为是在"拟合这个矩阵"
# 如0.6表示当前是fair时下次还是fair的概率
# 即：P(Y_{i}=Fair|Y_{i-1}=Fair)=0.6
#          2fair   2loaded    2start
# fair      0.6      0.4        0.0
# loaded    0.3      0.7        0.0
# start     0.5      0.5        0.0
# ###################################################
transition_mat = {'fair': np.array([0.6, 0.4, 0.0]),
                  'loaded': np.array([0.3, 0.7, 0.0]),
                  'start': np.array([0.5, 0.5, 0.0])}
states = list(transition_mat.keys())
state2ix = {'fair': 0,
            'loaded': 1,
            'start': 2}


# In[5]:


# ########################## 生成样本 ###################################
# 初始化为全零矩阵，然后填充，模拟出：sample_size 个序列 x 投掷 n_obs 次/序列
# rolls：5000个序列 x 每个序列投掷15次 x 每次是六选一[0,5]
#        六选一的概率由log_likelihood判断
# dices：5000个序列 x 每个序列投掷15次 x 每次是二选一{有偏、无偏}
#        依赖状态转移概率矩阵
# #####################################################################
def simulate_data(n_timesteps):
    data_list = np.zeros(n_timesteps)
    prev_state = 'start'
    state_list = np.zeros(n_timesteps)
    for n in range(n_timesteps):
        next_state = np.random.choice(states, p=transition_mat[prev_state])
        prev_state = next_state
        data_list[n] = np.random.choice([0, 1, 2, 3, 4, 5], p=probabilities[next_state])
        state_list[n] = state2ix[next_state]
    return data_list, state_list

sample_size = 10#5000  # 样本个数（或者说训练次数）
n_obs = 15  # 投掷次数
rolls_list = np.zeros((sample_size, n_obs)).astype(int) # 点数
status_list = np.zeros((sample_size, n_obs)).astype(int) # 骰子 {有偏、无偏}
for i in range(sample_size):
    rolls, dices = simulate_data(n_obs)
    rolls_list[i] = rolls.reshape(1, -1).astype(int)
    status_list[i] = dices.reshape(1, -1).astype(int)


# In[79]:


rolls_list.shape
status_list.shape


# # CRF_module模块

# In[ ]:


def crf_train_loop(model, rolls, targets, n_epochs, learning_rate=0.01):
    '''
    doc
    :param model: CRF
    :param rolls: 序列样本 | 骰子掷出的点数 5000x15， 或者句子里的词
    :param targets:  序列样本的标注 | 骰子的状态{有偏、无偏} 5000x15，或者词的BIOE标注
    :param n_epochs: 迭代轮数
    :param learning_rate:  学习率
    '''
    optimizer = Adam(model.parameters(), lr=learning_rate,
                     weight_decay=1e-4)
    for epoch in range(n_epochs):
        batch_loss = []
        N = rolls.shape[0]
        model.zero_grad()
        for index, (roll, labels) in enumerate(zip(rolls, targets)):
            # Forward Pass
            neg_log_likelihood = model.neg_log_likelihood(roll, labels)
            batch_loss.append(neg_log_likelihood)

            if index % 50 == 0: # batch_size=50
                ll = torch.cat(batch_loss).mean()
                ll.backward()
                optimizer.step()
                print("Epoch {}: Batch {}/{} loss is {:.4f}".format(epoch, index // 50, N // 50, ll.data.numpy()[0]))
                batch_loss = []
    return model

class CRF(torch.nn.Module):
    def __init__(self, n_dice, log_likelihood):
        super().__init__()
        self.n_states = n_dice
        self.loglikelihood = log_likelihood
        self.transition = torch.nn.init.normal(torch.nn.Parameter(torch.randn(n_dice, n_dice+1)), -1, 0.1)
        
    def to_scalar(self,var):
        return var.view(-1).data.tolist()[0]
    
    def argmax(self,vec):
        _, idx = torch.max(vec,1)
        return self.to_scalar(idx)
    
    def log_sum_exp(self,vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    
     def _data_to_likelihood(self, rolls):
        return Variable(torch.FloatTensor(self.loglikelihood[rolls]), requires_grad=False)


# # CRF训练及持久化

# In[ ]:


##
crf = CRF(2, log_likelihood)
model = crf_train_loop(crf, rolls, dices, 1, 0.001)
torch.save(model.state_dict(), "./checkpoint.hdf5")


# # CRF加载及使用

# In[ ]:


##
model.load_state_dict(torch.load("./checkpoint.hdf5"))
roll_list, dice_list = simulate_data(15)
test_rolls = roll_list.reshape(1, -1).astype(int)
test_targets = dice_list.reshape(1, -1).astype(int)
print(test_rolls[0])
print(model.forward(test_rolls[0])[0])
print(test_targets[0])
print(list(model.parameters())[0].data.numpy())


# In[148]:


log_likelihood.shape


# In[ ]:





# # 说明

# In[19]:


log_likelihood
rolls_list
status_list
n_dice = 2
n_states = n_dice
transition = torch.nn.init.normal(nn.Parameter(torch.randn(n_dice, n_dice + 1)), -1, 0.1)
transition
loglikelihoods = torch.FloatTensor(log_likelihood[rolls_list[0]])
loglikelihoods 


# ## neg_log_likelihood(self, rolls, states)

# In[15]:


for index, (rolls, states) in enumerate(zip(rolls_list, status_list)):
    if index==0:
        rolls
        states
        loglikelihoods = log_likelihood[rolls]
        states_ = torch.LongTensor(states)
        loglikelihoods.shape
        states_.shape


# ## \_compute\_likelihood_numerator(self,loglikelihoods,states)

# In[29]:


prev_state = 2
score = Variable(torch.Tensor([0]))
states_
transition
for index, state in enumerate(states_):
    print("---- at {}:".format(index))
    if index >= 0:
        print("state:{} prev_state:{} index:{} state:{} ".format(state, prev_state, index, state))
        print("transition[..]+loglikelihoods[..]={}+{}={}".format(transition[state, prev_state],loglikelihoods[index, state],transition[state, prev_state] + loglikelihoods[index, state]))
        score += transition[state, prev_state] + loglikelihoods[index, state]
        prev_state = state


# ## \_compute\_likelihood_denominator(self,loglikelihoods)
# $$alpha_t(j) = \sum_i alpha_{t-1}(i) * L(x_t | y_t) * C(y_t | y_{t-1} = i)$$
# 
# 这个描述的是在状态$y_t$下如何遍历地取所有可能的转移过来的概率
# 
# 所以这里三个乘法的前两项对当前状态来说是不变的，遍历$i$是更改的$C(..)$即转移概率
# 
# $alpha_{t-1}(i)$： 之前累乘（log下其实就是累加了）到当前的概率
# 
# $L(x_t | y_t)$：投掷到这个点数的概率
# 
# $C(y_t | y_{t-1} = i)$： 从状态$i$转移到当前状态$y_t$的概率

# In[66]:


def get_idx_of_dim1_max(vec):
    # torch.max 根据维度返回（该维度下最大值组成的tensor，索引）； |
    _, idx = torch.max(vec, dim=1)
    return idx.view(-1).data.tolist()[0]

def log_sum_exp(vec):
    a = vec[0, get_idx_of_dim1_max(vec)] # vec的 0行，max列（最大元素所在索引）
    a_broadcast = a.view(1, -1).expand(1, vec.size()[1])
    return a + torch.log(torch.sum(torch.exp(vec - a_broadcast)))


# In[61]:


n_states
transition
loglikelihoods
transition[:, n_states]
prev_alpha = transition[:, n_states] + loglikelihoods[0].view(1, -1)
prev_alpha


# In[60]:


roll = loglikelihoods[1]
next_state = 0
# next_state = 1

feature_function = transition[next_state, :n_states].view(1, -1) + roll[next_state].view(1,-1).expand(1,n_states)
feature_function


# In[62]:


prev_alpha
feature_function
alpha_t_next_state = prev_alpha + feature_function
alpha_t_next_state


# In[77]:


alpha_t_next_state
get_idx_of_dim1_max(alpha_t_next_state)
a = alpha_t_next_state[0, get_idx_of_dim1_max(alpha_t_next_state)] 

log_sum_exp(alpha_t_next_state)


# In[73]:


torch.max(torch.Tensor([[4,2]]), dim=1)


# # pyTorch

# In[152]:


help(torch.nn.init.normal_)


# ## 初始化
# ### torch.nn.init.normal
# 按指定的正态分布填充tensor
# ```
# >>> w = torch.empty(3, 5)
# >>> nn.init.normal_(w)
# ```
# ### torch.nn.Parameter
# Parameter会自动加入到Module的 .parameter 结果中，并且默认requires_grad=True
# 
# ### torch.randn
# 从标准正态分布(均值为0，方差为1）中生成随机数
# 

# In[26]:


###### 先定住随机数种子 #####
_ = torch.manual_seed(2019)


# In[27]:


class testA(nn.Module):
    def __init__(self):
        super().__init__()
#         # init.normal deprecated，改用init.normal_
#         self.transition2 = nn.init.normal(nn.Parameter(torch.randn(2,3)), -1, 0.1)
        self.transition = nn.Parameter(torch.Tensor(2,3))
        nn.init.normal_(self.transition)
        
testObj = testA()
testObj.transition


# ## torch.cat() 方法

# In[28]:





# ## .view() 方法 和 .expand()方法
# ### .view()
# - 类似resize，按view里给的n个参数表示shape
# - 特殊的： `.view(-1)` 就是展平的一维

# In[40]:


testObj.transition
print("view(-1):")
testObj.transition.view(-1)
print("view(1,-1)")
testObj.transition.view(1,-1)
print("view(1,-1).expand(1,2)")
testObj.transition.view(1,-1).shape
testObj.transition.view(1,-1).expand(2,6)


# ## .contiguous() 方法
# 这个方法就是重新拷贝一个tensor出来
# - 有这个的原因是因为，有些操作仅相当于改变了tensor的演示形状，比如 `narrow()，view()，expand()，transpose()`
# - 这些操作得到的tensor和原tensor是共享内存即共享data的
# - 比如把tensorA从(3,4)通过各种变换变成了(2,2,2)，此时修改shape(2,2,2)这个tensor中的某个值4变成-4，那么原tensor（shape(3,4)）中的4也变成了-4，这类操作有一定好处，比如变换成更容易理解的维度再修改某个值，这样可能带来一定的物理意义之类的

# ## .data 属性

# In[173]:


testObj.transition.view(-1).data
testObj.transition.view(-1).data.tolist()
testObj.transition.view(-1).tolist()


# ## torch.max

# In[215]:


get_ipython().run_line_magic('pinfo', 'torch.max')


# In[248]:


a=torch.Tensor([[3,2,4],
                [9,3,1]])
a
a[0][0],a[1][0]
res = a.tolist()
res1 = [res[0],res[1]]
res2 = [res1[0],res[1]]
torch.max(a,dim=0)


# In[75]:


a = torch.Tensor([[[10,20],[3,4],[5,6]],[[1,40],[60,40],[50,60]]])
a
a.shape

"---dim=1---"
torch.max(a,dim=1)[0]
[(a[0,0,0],a[0,1,0],a[0,2,0]),(a[0,0,1],a[0,1,1],a[0,2,1])]
[(a[1,0,0],a[1,1,0],a[1,2,0]),(a[1,0,1],a[1,1,1],a[1,2,1])]

"---dim=0---"
torch.max(a,dim=0)[0]
[(a[0,0,0],a[1,0,0]),(a[0,0,1],a[1,0,1])]
[(a[0,1,0],a[1,1,0]),(a[0,1,1],a[1,1,1])]
[(a[0,2,0],a[1,2,0]),(a[0,2,1],a[1,2,1])]
# a[0,0,0],a[1,0,0]
# a[0,0,1],a[1,0,1]

# a[0,0],a[1,0]
# a[0,1],a[1,1]
# a[0,2],a[1,2]


# ## torch.FloatTensor(xxxx, requires_grad=False)
# `requires_grad` 默认是`False`，如果被设置为`True`则该节点（tensor）以及所有依赖该节点（tensor）

# In[130]:


log_likelihood
rolls
log_likelihood.shape
rolls.shape
log_likelihood[rolls].shape

ft = torch.FloatTensor(log_likelihood[rolls])
ft[0][0]


# ## torch.cat(loss).mean()

# ## torch.cat(loss).mean().backward()

# ## optimizer.step()

# ## var.view(-1).data.tolist()[0]

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




