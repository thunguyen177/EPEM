from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
import numpy as np
import impyute as impy
from fancyimpute import IterativeSVD, SoftImpute, NuclearNormMinimization
import pandas as pd
import time




"""### LDA and nan function"""

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

"""### compute_err function"""

def missing_rate(Xtrain, ytrain, n, p, G):    
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


def compute_err_Nuclear(Xtrain, ytrain, Xtest, ytest, n, p, G):    
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

    scaler = MinMaxScaler()
    scaler.fit(Xtr_nan)
    Xtr_nan = scaler.transform(Xtr_nan)
    Xtest = scaler.transform(Xtest)
    Xtr_nan_list2 = []
    for g in range(G):
      Xtr_nan_list2.append(scaler.transform(Xtr_nan_list[g]))
    
    #impute,classify and get the error rates for imputation approaches    
    start = time.time()
    Xtr_nuclear = NuclearNormMinimization(max_iters=10).fit_transform(Xtr_nan)
    clf_nuclear = skLDA().fit(Xtr_nuclear, ytr)
    nuclear_err = np.mean(clf_nuclear.predict(Xtest).flatten() != ytest)
    nuclear_time = time.time()-start
 
    return nuclear_err, nuclear_time

"""## Import Fashion MNIST"""

import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(Xtrain, ytrain), (Xtest, ytest) = fashion_mnist.load_data()

Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape

Xtrain = Xtrain.astype(float).reshape((60000,784))

# set random seed and shuffle the data
np.random.seed(1)
idx = np.arange(len(ytrain))
np.random.shuffle(idx)
Xtrain, ytrain = Xtrain[idx,:], ytrain[idx]  

Xtrain.shape, ytrain.shape

# convert the test set to NumPy arrays and flatten the data
Xtest = Xtest.astype(float).reshape((10000,784))

# number of sample per class in training data
ng = np.asarray([sum(ytrain==i) for i in np.arange(10)])

"""## 20%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4500,4200,4000, 3800],
                                 10).reshape((10,-1))))
p = np.array([310,400,480, 520,784])   
missing_rate(Xtrain, ytrain, n, p, 10)

nuclear20 = compute_err_Nuclear(Xtrain, ytrain, Xtest, ytest, n, p, 10)

"""## 30%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4400,4000,3400, 3000],
                                 10).reshape((10,-1))))
p = np.array([250,310,400, 450,784])   
missing_rate(Xtrain, ytrain, n, p, 10)

nuclear30 = compute_err_Nuclear(Xtrain, ytrain, Xtest, ytest, n, p, 10)

"""## 40%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([3600,3400,3000, 2500],
                                 10).reshape((10,-1))))
p = np.array([200,220,300, 400,784])   
missing_rate(Xtrain, ytrain, n, p, 10)

nuclear40 = compute_err_Nuclear(Xtrain, ytrain, Xtest, ytest, n, p, 10)

"""## 50%"""
n = np.hstack((ng.reshape((-1,1)), np.tile([3000,2900,2700, 2500],
                                 10).reshape((10,-1))))
p = np.array([100,150,220, 250,784])   
missing_rate(Xtrain, ytrain, n, p, 10)


result = np.vstack((nuclear20, nuclear30, nuclear40))
np.savetxt("fashion_nuclear.csv", result, delimiter=",")

