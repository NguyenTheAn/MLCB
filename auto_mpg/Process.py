#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import random


# In[25]:


np.random.seed(5)
df = pd.read_csv("auto_mpg_dataset.csv")
Mean = df['horsepower'][df['horsepower'] != -100000].mean()
df['horsepower'] = df['horsepower'].replace(-100000, Mean)
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']].values
Y = df['mpg'].values


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_test = Y_test.reshape(Y_test.shape[0], 1)


# In[27]:


one1 = np.ones((X_train.shape[0], 1))
Xbar_train = np.concatenate((one1, X_train), axis = 1)


# In[28]:


A = np.dot(Xbar_train.T, Xbar_train)
B = np.dot(Xbar_train.T, Y_train)
result = np.dot(np.linalg.pinv(A), B)


# In[29]:


one2 = np.ones((X_test.shape[0], 1))
Xbar_test = np.concatenate((one2, X_test), axis = 1)


# In[30]:


loss = (1/(2*Xbar_test.shape[0])) * (np.linalg.norm(Xbar_test.dot(result) - Y_test)**2)
print("Loss on testset: {}".format(loss))


# In[32]:


from tqdm import tqdm

def grad(w):
    N = Xbar_train.shape[0]
    return (1/N) * Xbar_train.T.dot(Xbar_train.dot(w) - Y_train)

w = [np.random.rand(8, 1)]
lr = 1e-7
L = []

for i in tqdm(range(500000)):
    loss = (1/(2*Xbar_train.shape[0])) * (np.linalg.norm(Xbar_train.dot(w[-1]) - Y_train)**2)
    L.append(loss)
#     p = np.random.permutation(len(Xbar_train))
#     Xbar_train = Xbar_train[p]
#     Y_train = Y_train[p]
    w_new = w[-1] - lr * grad(w[-1])
    if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
        break
    w.append(w_new)

loss = (1/(2*Xbar_test.shape[0])) * (np.linalg.norm(Xbar_test.dot(w[-1]) - Y_test)**2)
print("Loss on testset: {}".format(loss))


# In[34]:


import matplotlib.pyplot as plt
plt.plot(L)
plt.show()


# In[35]:


reg = linear_model.LinearRegression();
reg.fit(Xbar_train, Y_train)


# In[36]:


y_pred = reg.predict(Xbar_test)
loss = (1/(2*Xbar_test.shape[0])) * (np.linalg.norm(y_pred - Y_test)**2)
print("Loss on testset: {}".format(loss))


# In[ ]:





# In[ ]:




