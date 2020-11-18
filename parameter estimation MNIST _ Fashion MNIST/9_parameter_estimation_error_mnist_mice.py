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

def compute_err_mice(Xtrain, ytrain, n, p, G):      
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
    Xtr_mice = IterativeImputer(max_iter=100).fit(Xtr_nan).transform(Xtr_nan)
    mus_mice = np.asarray([np.mean(Xtr_mice[ytrain==g,:], axis=0
                                   ) for g in np.arange(G)])
    S_mice = np.asarray([(sum(ytrain==g)-1)*np.cov(Xtr_mice[ytrain==g,:], rowvar =False) 
             for g in np.arange(G)])
    S_mice = S_mice/len(ytrain)
    mice_err = err(mus, S, mus_mice, S_mice)
    mice_time = time.time()-start

    return mice_err, mice_time, per_missing


"""# Import MNIST"""

import tensorflow as tf
import tensorflow_datasets as tfds

# Fetch the dataset directly
mnist = tfds.image.MNIST()
# or by string name
mnist = tfds.builder('mnist')

# Download the data, prepare it, and write it to disk
mnist.download_and_prepare()

# Load data from disk as tf.data.Datasets
datasets = mnist.as_dataset()
train_dataset, test_dataset = datasets['train'], datasets['test']

# convert the Dataset to NumPy arrays and flatten the data
Xtrain, ytrain = [], []
for example in tfds.as_numpy(train_dataset):
  Xtrain.append(example['image'].flatten())
  ytrain.append(example['label'])

Xtrain, ytrain = np.asarray(Xtrain), np.asarray(ytrain)
Xtrain = Xtrain.astype(float)

# convert the test set to NumPy arrays and flatten the data
Xtest, ytest = [], []
for example in tfds.as_numpy(test_dataset):
  Xtest.append(example['image'].flatten())
  ytest.append(example['label'])

Xtest, ytest = np.asarray(Xtest), np.asarray(ytest)
Xtest = Xtest.astype(float)

X = np.vstack((Xtrain,Xtest))
y = np.hstack((ytrain,ytest))
X.shape, y.shape

# set random seed and shuffle the data
np.random.seed(1)
idx = np.arange(len(y))
np.random.shuffle(idx)
X, y = X[idx,:], y[idx]  

X.shape, y.shape

# check if a column is all 0
id = [np.sum(Xtrain[:,i] != 0)>10 for i in range(28**2)]
# number of columns that mostly zero
print(28**2-np.sum(id))
# number of columns that has at least more than 10 non-zero
np.sum(id)

X = X[:, id]

# number of sample per class in training data
ng = np.asarray([sum(ytrain==i) for i in np.arange(10)])
ng

"""### 20%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4500,4000,3200, 3000],
                                 10).reshape((10,-1))))
p = np.array([300,320,400, 500,649]) 
missing_rate(X, y, n, p, 10)

mice20 = compute_err_mice(X, y, n, p, 10) 

"""### 30%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4000,3500,3200, 3000],
                                 10).reshape((10,-1))))
p = np.array([150,270,300, 370,649]) 
missing_rate(X, y, n, p, 10)

mice30 = compute_err_mice(X, y, n, p, 10)  
"""## 40%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([3500,3000,2500, 2100],
                                 10).reshape((10,-1))))
p = np.array([150,200,250, 300,649]) 
missing_rate(X, y, n, p, 10)

mice40 = compute_err_mice(X, y, n, p, 10) 

"""## 50%"""
n = np.hstack((ng.reshape((-1,1)), np.tile([2300,2100,2000, 1900],
                                 10).reshape((10,-1))))
p = np.array([90,110,120, 140,649]) 
missing_rate(X, y, n, p, 10)


result = np.vstack((mice20, mice30, mice40))
np.savetxt("mnist_para_estimate_mice.csv", result, delimiter=",")
