#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# In[16]:


my_data=pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv")


# In[23]:


def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    return np.asarray(target)
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values
x=removeColumns(my_data,0,1)
y=target(my_data,1)


# In[38]:


from sklearn.model_selection import train_test_split
x_trainset,x_testset,y_trainset,y_testset=train_test_split(x,y,test_size=0.3,random_state=7)


# In[40]:


print(x_trainset.shape)
print(y_trainset.shape)
print(x_testset.shape)
print(y_testset.shape)


# In[43]:


neigh=KNeighborsClassifier(n_neighbors=1)
neigh23=KNeighborsClassifier(n_neighbors=23)
neigh90=KNeighborsClassifier(n_neighbors=90)


# In[44]:


neigh.fit(x_trainset,y_trainset)
neigh23.fit(x_trainset,y_trainset)
neigh90.fit(x_trainset,y_trainset)


# In[49]:


pred=neigh.predict(x_testset)
pred23=neigh23.predict(x_testset)
pred90=neigh90.predict(x_testset)
print(pred)
print(pred23)
print(pred90)


# In[51]:


from sklearn import metrics
print(("Neigh's Accuracy: "),metrics.accuracy_score(y_testset),pred)
print(("Neigh's Accuracy: "),metrics.accuracy_score(y_testset),pred23)
print(("Neigh's Accuracy: "),metrics.accuracy_score(y_testset),pred90)


# In[ ]:




