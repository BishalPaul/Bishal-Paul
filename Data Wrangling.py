#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


filename="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"


# In[4]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
"drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
"num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
"peak-rpm","city-mpg","highway-mpg","price"]
df=pd.read_csv(filename,names=headers)
print("Done")


# In[6]:


df.head()


# In[7]:


import numpy as np


# In[10]:


#replace"?" to NaN(not a number)
df.replace("?",np.nan,inplace=True)
df.head(5)


# In[11]:


missing_data=df.isnull()
missing_data.head(5)


# In[12]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[16]:


avg1=df["normalized-losses"].astype("float").mean(axis=0)
avg1


# In[18]:


df["normalized-losses"].replace(np.nan,avg1,inplace=True)
df.head(5)


# In[19]:


avg2=df['bore'].astype('float').mean(axis=0)
avg2


# In[20]:


df['bore'].replace(np.nan,avg2,inplace=True)
df.head(5)


# In[21]:


avg4=df["horsepower"].astype("float").mean(axis=0)
avg4


# In[23]:


df['horsepower'].replace(np.nan,avg4,inplace=True)
df.head(5)


# In[24]:


avg5=df["stroke"].astype("float").mean(axis=0)
avg5


# In[26]:


df['stroke'].replace(np.nan,avg4,inplace=True)
df.tail(5)


# In[27]:


df['num-of-doors'].value_counts()


# In[31]:


df["num-of-doors"].value_counts().idxmax()


# In[32]:


df['num-of-doors'].replace(np.nan,"four",inplace=True)
df.head(5)


# In[33]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print("Done")


# In[34]:


df.head(5)


# In[ ]:




