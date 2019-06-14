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


text = ""
sensitiveWords = {
    1:["a","i"],
    2:["b","m"],
    3:["c"],
#     4:[""], ...
}
inp = sensitiveWords


# In[5]:


[k for k,v in inp.items() if "a" in v]


# In[22]:


a = [1,2,3]
b = a.copy()
b.append(1)
a
b


# In[15]:



class Node(object):  
    def __init__(self):  
        self.children = None  

def add_word(root,word):  
    node = root  
    for i in range(len(word)):  
        if node.children == None:  
            node.children = {}  
            node.children[word[i]] = Node()  
  
        elif word[i] not in node.children:  
            node.children[word[i]] = Node()  
        node = node.children[word[i]]

def init(path):  
    root = Node()  
    fp = open(path,'r')
    for line in fp:  
        line = line[0:-1]  
        #print len(line)  
        #print line  
        #print type(line)  
        add_word(root,line)  
    fp.close()  
    return root  


def is_contain(message, root):  
    for i in range(len(message)):  
        p = root  
        j = i  
        while (j<len(message) and p.children!=None and message[j] in p.children):  
            p = p.children[message[j]]  
            j = j + 1  
  
        if p.children==None:  
            #print '---word---',message[i:j]  
            return True  
      
    return False  
  


# In[91]:


from itertools import islice,takewhile
text_inp = "come from chian-town and has been study math for 12 years"
textIter = iter(" " + text_inp + " ")
next(islice(textIter,39,40))
next(islice(textIter,40,41))


# In[47]:


from itertools import islice,takewhile
import re

text_inp = "come,   from,chian-town    and,as been study math for 12 years"
textIter = iter(" " + text_inp + " ")
text_inp
re.sub("[^\\w]+"," ",text_inp)
text_inp


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


m = None
m is None
m == None


# In[67]:


from collections import deque
import re

class Node(object):
    """
    DFA不需要反查,所以未记录父节点
    在DFA中叶子节点标记了一个词的结束如, "porn "和"porno"中的" "和"o"
    """
    def __init__(self, reason=None):
        self.children = None
        self.reason = reason # 仅叶子节点有值, 标记某个词是什么原因被标记为敏感词

    def is_leaf_node(self):
        return self.children is None

    def add_child(self,v):
        if self.is_leaf_node():
            self.children = {v:Node()}
        elif v not in self.children:
            self.children[v] = Node()
        else:
            pass # DFA子节点已有则不更新



class DFATree(object):
    def __init__(self):
        self.root = Node()

    def add_word(self,word_inp,reason,sep = " "):
        word = sep+word_inp+sep
        node = self.root
        for idx,char in enumerate(word):
            node.add_child(char)
            node = node.children[char]
        node.reason = reason

    def add_word_CN(self,word_inp,reason):
        self.add_word(word_inp,reason,sep="")

    def add_word_EN(self,word_inp,reason):
        self.add_word(word_inp,reason,sep=" ")

    def init_from_dict(self,watch_list_inp:dict):
        for reason,word_list in watch_list_inp.items():
            for word in word_list:
                self.add_word_EN(word,reason)

    def search_old(self, text_inp:str):
        textIter = iter(" " + text_inp + " ")
        for idx,char in enumerate(textIter):
            p = self.root
            print("\nroot-p: ",p)
            print("root-p.children: ",p.children)
            print("root-p.reason: ",p.reason)
            print("message[i] ", char)
                
            while j < len(text) and p.children is not None and text[j] in p.children:
                print(" ")
                print("  message[j] ",text[j])
                p = p.children[text[j]]
                print("  child-p: ",p)
                print("  child-p.children: ",p.children)
                print("  child-p.reason: ",p.reason)
                j += 1
            if p.children is None:
                return  True,p.reason
        return False,-1
    
    def search(self, text_inp:str):
        text = " " + re.sub("[^\\w]+"," ",text_inp) + " "
        word_list = deque()
        for idx,char in enumerate(text):
            p = self.root
#             print("\nroot-p: ",p)
#             print("root-p.children: ",p.children)
#             print("root-p.reason: ",p.reason)
            if char in self.root.children:
                j = idx
                p = self.root
                print(j,p,text,p.children,j<len(text),text[j] in p.children)
                while j<len(text) and text[j] in p.children:
                    print("b"," "*j,j,"'"+text[j]+"'")
                    p = p.children[text[j]]
                    if p.children is None:
                        word_list.append(("".join(text[idx:j]).strip(),p.reason))
                        pass # 这里不用跳跃性地赋值 idx = idx+j,因为要兼容类似中文的语种,从 "我爱天安门" 中找到"我爱"和"爱天安门"
                    j += 1
                    print("e"," "*(j-1),j,"'"+text[j]+"'",p.children,p)
        return word_list

dfa = DFATree()

watch_list = {0:["massacre", "porn", "violence", "kill", "violate purpose"]}
dfa.init_from_dict(watch_list)


# In[76]:


text = " " + re.sub("[^\\w]+"," ",text_inp) + " "
text[16:22]
text[22]
p = dfa.root.children[' '].children['v'].children['i'].children['o'].children['l'].children['a'].children['t'].children['e']
p
p = p.children[text[22]]
p
p.children
p.children is None


# In[71]:





# In[77]:


text_inp = "massacreadwgb violate"
# res = dfa.search("massacreadwgb violate, purpose, violate purpose, from now on massacre")
res = dfa.search(text_inp)
print(res,"\n"*10)


# In[ ]:





# In[ ]:





# In[62]:


def loop(dict_inp,word_list):
    for k in dict_inp.keys():
        word_list.append(k)
        if dict_inp[k].children is None:
            print(k)
        else:
            word_list.extend(loop(dict_inp[k],word_list))
    return word_list

the_dict = dfa.root.children
for i in the_dict.keys():
    the_dict[i].children
the_dict[' '].children
the_dict[' '].children['p'].children
the_dict[' '].children['p'].children['o'].children
the_dict[' '].children['p'].children['o'].children['r'].children
the_dict[' '].children['p'].children['o'].children['r'].children['n'].children
the_dict[' '].children['p'].children['o'].children['r'].children['n'].children[' '].children is None

loop(the_dict,[])


# In[ ]:


k

