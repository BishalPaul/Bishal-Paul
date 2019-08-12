#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#diabetes=load_diabetes()
#print(diabetes)


# In[4]:


diabetes_x=diabetes.data[:,None,2]
diabetes_x


# In[5]:


LinReg=LinearRegression()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_trainset,x_testset,y_trainset,y_testset=train_test_split(diabetes_x,diabetes.target,test_size=0.3,random_state=7)lin


# In[9]:


LinReg.fit(x_trainset,y_trainset)


# In[11]:


LinReg.intercept_


# In[12]:


LinReg.coef_


# In[19]:


plt.scatter(x_testset,y_testset,color='red',linewidth=3)
plt.plot(x_testset,LinReg.predict(x_testset),color='yellow',linewidth=3)


# In[20]:





# In[ ]:




