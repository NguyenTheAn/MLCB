#!/usr/bin/env python
# coding: utf-8

# In[231]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# In[232]:


N = 500
np.random.seed(1)
means = np.array([[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]])
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)
X = np.concatenate((X0, X1, X2, X3, X4))
Y = np.asarray([[0]*N + [1]*N + [2]*N + [3]*N + [4]*N]).T
Y = np_utils.to_categorical(Y, 5)
X_test = np.random.multivariate_normal(means[0], cov, 50)
X_test = np.concatenate((X_test, np.random.multivariate_normal(means[1], cov, 50)))
X_test = np.concatenate((X_test, np.random.multivariate_normal(means[2], cov, 50)))
X_test = np.concatenate((X_test, np.random.multivariate_normal(means[3], cov, 50)))
X_test = np.concatenate((X_test, np.random.multivariate_normal(means[4], cov, 50)))
Y_test = np.asarray([[0]*50 + [1]*50 + [2]*50 + [3]*50 + [4]*50]).T
Y_test = np_utils.to_categorical(Y_test, 5)
plt.plot(X[:, 0], X[:, 1], 'o')
plt.plot(X_test[:, 0], X_test[:, 1], 'ro')
plt.show()


# In[233]:


def init_centroids(cluster, X):
    centroids = X[np.random.choice(X.shape[0], cluster)]
    return centroids

def cdist(X, centroids):
    dist = np.zeros((X.shape[0], 5))
    for i in range(len(X)):
        tmp = np.array([X[i]]*5)
        dist[i] = np.linalg.norm((tmp-centroids), axis = 1)
    return dist

def assign_labels(centroids, X):
    D = cdist(X, centroids)
    return np.argmin(D, axis = 1)

def update_centroids(cluster, labels, X):
    centroids = np.zeros((cluster, X.shape[1]))
    for k in range(cluster):
        Xk = X[labels == k, :]
        centroids[k,:] = np.mean(Xk, axis = 0) 
    return centroids

def check(centroids, new_centroids):
    cmp = centroids == new_centroids
    return len(cmp[cmp==True]) == (centroids.shape[0] * centroids.shape[1])


# In[234]:


def Kmean(cluster, X):
    centroids = [init_centroids(cluster, X)]
    while True:
        labels = assign_labels(centroids[-1], X)
        new_centroids = update_centroids(cluster, labels, X)
        if check(centroids[-1], new_centroids) == True:
            centroids.append(new_centroids)
            break;
        centroids.append(new_centroids)
    return centroids 


# In[235]:


cluster = 5
centroids = Kmean(cluster, X)


# In[236]:


print(centroids[-1])
len(centroids)
# means = np.array([[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]])


# In[206]:




