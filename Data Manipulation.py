#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
csv_path='https://ibm.box.com/shared/static/keo2qz0bvh4iu6gf5qjq4vdrkt67bvvb.csv'
df=pd.read_csv(csv_path)
print("Done")


# In[2]:


df.head()


# In[3]:


df.tail()


# In[8]:


x=df[["Length"]]
print(x)


# In[10]:


y=df[["Artist","Length","Album"]]
print(y)


# In[ ]:




