#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[5]:


path="https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv"
df=pd.read_csv(path)
df.head(5)


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


lm=LinearRegression()
lm


# In[9]:


x=df[["highway-mpg"]]
y=df['price']


# In[10]:


lm.fit(x,y)


# In[14]:


yha=lm.predict(x)
yha[0:5]


# In[12]:


lm.intercept_


# In[13]:


lm.coef_


# In[15]:


z=df[['horsepower','curb-weight','engine-size','highway-mpg']]


# In[16]:


lm.fit(z,y)


# In[18]:


ar=lm.predict(z)
ar[0:5]


# In[20]:


lm.intercept_


# In[21]:


lm.coef_


# In[ ]:




