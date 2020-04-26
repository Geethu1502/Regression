#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[5]:


# understanding the dataset
boston= load_boston()
print(boston.DESCR)


# In[6]:


# access data attributes 
dataset=boston.data
for name, index in enumerate(boston.feature_names):
    print(index,name)


# In[11]:


# reshaping data
data= dataset[:,12].reshape(-1,1)


# In[8]:


# shape of the data
np.shape(dataset)


# In[9]:


# target values
target=boston.target.reshape(-1,1)


# In[10]:


# shape of the target
np.shape(target)


# In[13]:


# ensuring that matplotlib is working
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='blue')
plt.xlabel('lower income population')
plt.ylabel('cost of house')
plt.show()


# In[15]:


# regression
from sklearn.linear_model import LinearRegression

#creating a regression model
reg=LinearRegression()

# fit the model:
reg.fit(data,target)


# In[16]:


# prediction
pred=reg.predict(data)


# In[18]:


# ensuring that matplotlib is working
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='green')
plt.plot(data,pred,color='red')
plt.xlabel('lower income population')
plt.ylabel('cost of house')
plt.show()


# In[19]:


# circumventing curve issue using polynomial model
from sklearn.preprocessing import PolynomialFeatures

# to allow merging of models
from sklearn.pipeline import make_pipeline


# In[21]:


model=make_pipeline(PolynomialFeatures(3), reg)


# In[22]:


model.fit(data,target)


# In[23]:


pred=model.predict(data)


# In[24]:


# ensuring that matplotlib is working
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='green')
plt.plot(data,pred,color='red')
plt.xlabel('lower income population')
plt.ylabel('cost of house')
plt.show()


# In[25]:


# r_2 metric
from sklearn.metrics import r2_score


# In[26]:


# predict
r2_score(pred,target)


# In[ ]:




