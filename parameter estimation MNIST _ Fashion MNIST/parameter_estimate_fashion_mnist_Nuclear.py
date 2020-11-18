from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
import numpy as np
import impyute as impy
from fancyimpute import IterativeSVD, SoftImpute, NuclearNormMinimization
import pandas as pd
import time


"""### nan function"""

'''
function that create data list that contain missing values
The input X is a numpy array, y is the label
the function return a list where the ith element of 
the list belongs to the ith class
'''

def make_nan_list(X,y,G, n, p):
    # note that the label should go from 0 to G-1
    data = []
    for g in np.arange(G):
        data.append(X[y==g,:])
        for k in np.arange(len(p)-1):
            data[g][n[g,k+1]:n[g,k], p[k]:] = np.nan
    return data

def missing_rate(Xtrain, ytrain, n, p, G):
    # function that compute the missing rate of a given pattern    
    Xtr_nan_list = make_nan_list(Xtrain,ytrain,G, n, p)
    # make NA data
    # since making function changes the order of observation
    # we need to generate new ytr from Xtr_nan    
    Xtr_nan, ytr = Xtr_nan_list[0], np.repeat(0, len(Xtr_nan_list[0]))
    for g in np.arange(1,G):
        Xtr_nan = np.vstack((Xtr_nan, Xtr_nan_list[g]))
        ytr = np.hstack((ytr, np.repeat(g, len(Xtr_nan_list[g]))))

    # percentage of missing values
    per_missing = np.mean(np.isnan(Xtr_nan))
    return per_missing

"""### compute_err function"""

def err(mus, S, mus_est, S_est):
  err_rate = (np.linalg.norm(mus_est-mus))/mus.size 
  err_rate += (np.linalg.norm(S_est-S))/S.size 
  return err_rate

def compute_err_Nuclear(Xtrain, ytrain, n, p, G):      
    # make NAs
    Xtr_nan_list = make_nan_list(Xtrain,ytrain,G, n, p)
    # make NA data
    # since making function changes the order of observation
    # we need to generate new ytr from Xtr_nan    
    Xtr_nan, ytr = Xtr_nan_list[0], np.repeat(0, len(Xtr_nan_list[0]))
    for g in np.arange(1,G):
        Xtr_nan = np.vstack((Xtr_nan, Xtr_nan_list[g]))
        ytr = np.hstack((ytr, np.repeat(g, len(Xtr_nan_list[g]))))

    scaler = StandardScaler()
    scaler.fit(Xtr_nan)
    Xtr_nan = scaler.transform(Xtr_nan)
    Xtrain = scaler.transform(Xtrain)
    for g in range(G):
      Xtr_nan_list[g] = scaler.transform(Xtr_nan_list[g])

    mus = [np.mean(Xtrain[ytrain==g,:], axis=0) for g in np.arange(G)]
    mus = np.asarray(mus) # each row is a mean of a class
    S = [(sum(ytrain==g)-1)*np.cov(Xtrain[ytrain==g,:],rowvar =False) 
             for g in np.arange(G)]
    S = np.asarray(S)/len(ytrain)

    # percentage of missing values
    per_missing = np.mean(np.isnan(Xtr_nan))

    start = time.time()
    Xtr_nuclear = NuclearNormMinimization(max_iters=100).fit_transform(Xtr_nan)
    mus_nuclear = np.asarray([np.mean(Xtr_nuclear[ytrain==g,:], axis=0
                                      ) for g in np.arange(G)])
    S_nuclear = np.asarray([(sum(ytrain==g)-1)*np.cov(Xtr_nuclear[ytrain==g,:], rowvar =False) 
             for g in np.arange(G)])
    S_nuclear = S_nuclear/len(ytrain)
    nuclear_err = err(mus, S, mus_nuclear, S_nuclear)
    nuclear_time = time.time()-start 

    return nuclear_err, nuclear_time, per_missing

"""# Import MNIST"""

import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(Xtrain, ytrain), (Xtest, ytest) = fashion_mnist.load_data()

Xtrain = Xtrain.astype(float).reshape((60000,784))

# set random seed and shuffle the data
np.random.seed(1)
idx = np.arange(len(ytrain))
np.random.shuffle(idx)
Xtrain, ytrain = Xtrain[idx,:], ytrain[idx]  

Xtrain.shape, ytrain.shape

# convert the test set to NumPy arrays and flatten the data
Xtest = Xtest.astype(float).reshape((10000,784))

X = np.vstack((Xtrain, Xtest))
y = np.hstack((ytrain, ytest))

# number of sample per class in training data
ng = np.asarray([sum(y==i) for i in np.arange(10)])
ng

"""### 20%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([5000,4500,4200, 4000],
                                 10).reshape((10,-1))))
p = np.array([380,450,500, 520,784])   
missing_rate(X, y, n, p, 10)

nuc20 = compute_err_Nuclear(X, y, n, p, 10) 


"""### 30%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4000,3500,3000, 2800],
                                 10).reshape((10,-1))))
p = np.array([350,450,500, 520,784])   
missing_rate(X, y, n, p, 10)


nuc30 = compute_err_Nuclear(X, y, n, p, 10) 

"""## 40%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4500,3490,3350, 3100],
                                 10).reshape((10,-1))))
p = np.array([180,250,400, 450,784])   
missing_rate(X, y, n, p, 10)

nuc40 = compute_err_Nuclear(X, y, n, p, 10) 
"""## 50%"""
n = np.hstack((ng.reshape((-1,1)), np.tile([3500,3290,3100, 3000],
                                 10).reshape((10,-1))))
p = np.array([90,110,200, 250,784])   
missing_rate(X, y, n, p, 10)

result = np.vstack((nuc20, nuc30, nuc40))
np.savetxt("fashion_parameter_est_Nuclear.csv", result, delimiter=",")


