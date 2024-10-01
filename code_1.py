#!/usr/bin/env python
# coding: utf-8

# In[2]:


# loading required libraries
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# reading input data
data = pd.read_csv('data2023.csv')
data.head()


# In[4]:


data.info()


# In[5]:


# cosine similarity function

# using simple calculations
def dot(v1,v2):
    total=0
    for i in range(0,len(v1)):
        total+=v1[i]*v2[i]
        
    return total

# using dot product function
def cosine(v1,v2):
    
    return dot(v1,v2)/math.sqrt(dot(v1,v1)*dot(v2,v2))

# using numpy version of dot product function
def cosine_np(v1,v2):
    return np.dot(v1,v2)/np.sqrt(np.dot(v1,v1)*np.dot(v2,v2))


# In[8]:


# one pair similarity measurements

# simple cosine function
def one_pair_cosine_similarity(docmatrix):
    docA = docmatrix[:,0]
    docB = docmatrix[:,1]
    sim = cosine(docA,docB)
    return sim

# numpy cosine function
def one_pair_cosine_np_similarity(docmatrix):
    docA = docmatrix[:,0]
    docB = docmatrix[:,1]
    sim = cosine_np(docA,docB)
    return sim


# In[9]:


# all pairs similarity measurements

# simple methods
def all_pairs_cosine_similarity(docmatrix):
    sim_vec = []
    docvectors = np.transpose(docmatrix)
    for docA in docvectors:
        for docB in docvectors:
            sim_vec.append(cosine(docA,docB))
    return sim_vec

# numpy methods
def all_pairs_cosine_np_similarity(docmatrix):
    sim_vec = []
    docvectors = np.transpose(docmatrix)
    for docA in docvectors:
        for docB in docvectors:
            sim_vec.append(cosine_np(docA,docB))
    return sim_vec


# In[10]:


# algorithm mean time function
def timeit(somefunc,*args,repeats=10,**kwargs):
    times=[]
  
    while repeats>0:
        starttime=time.time()
        ans=somefunc(*args,**kwargs)
        endtime=time.time()
        timetaken=endtime-starttime
        times.append(timetaken)
        repeats-=1
    
    mean=np.mean(times)
    stdev=np.std(times)
 
    return (mean,stdev)


# In[11]:


# one pair cosine similarity analysis
xs1 = []
ys1 = []
for n in range(1000,21155,1000):
    docmatrix = data.iloc[:n,1:3].to_numpy()
    (mean,stdev)=timeit(one_pair_cosine_similarity,docmatrix,repeats=20)
    xs1.append(n)
    ys1.append(mean)
    
xs2 = []
ys2 = []
for n in range(1000,21155,1000):
    docmatrix = data.iloc[:n,1:3].to_numpy()
    (mean,stdev)=timeit(one_pair_cosine_np_similarity,docmatrix,repeats=100)
    xs2.append(n)
    ys2.append(mean)
    
fig,ax = plt.subplots(1,2,figsize = [12,5])
ax[0].plot(xs1,ys1)
ax[0].set_xlabel('document dimension')
ax[0].set_ylabel('one pair similarity time')
ax[0].set_title('one pair cosine similarity')

ax[1].plot(xs2,ys2)
ax[1].set_xlabel('document dimension')
ax[1].set_ylabel('one pair similarity time')
ax[1].set_title('one pair cosine similarity with numpy')


# In[12]:


# Jaccard similarity function
def jaccard(doc1,doc2):
    intersection = []
    for i in range(len(doc1)):
        intersection.append(min(doc1[i],doc2[i]))
    union = np.sum(doc1)+np.sum(doc2)-np.sum(intersection)
    return np.sum(intersection)/union


# In[13]:


# one pair jaccard similarity analysis
xs1 = []
ys1 = []
for n in range(1000,21155,1000):
    docmatrix = data.iloc[:n,1:3].to_numpy()
    (mean,stdev)=timeit(jaccard,docmatrix[:,0],docmatrix[:,1],repeats=50)
    xs1.append(n)
    ys1.append(mean)
        
fig,ax = plt.subplots(figsize = [7,5])
ax.plot(xs1,ys1)
ax.set_xlabel('document dimension')
ax.set_ylabel('one pair similarity time')
ax.set_title('one pair time complexity')


# In[ ]:




