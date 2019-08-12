#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import pandas library
import pandas as pd
path="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df=pd.read_csv(path,header=None)
print("Done")


# In[4]:


df.head()


# In[5]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
"drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
"num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
"peak-rpm","city-mpg","highway-mpg","price"]


# In[8]:


df.columns=headers
df.head(4)


# In[9]:


df.to_csv("automobile.csv")


# In[10]:


df.dtypes


# In[13]:


df.describe(include="all")


# In[16]:



#look at the info of df
df.info


# In[ ]:




