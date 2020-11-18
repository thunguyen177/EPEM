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

def compute_err_EM(Xtrain, ytrain, Xtest, ytest, n, p, G):    
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
    Xtr_em = impy.em(Xtr_nan,loops=10)
    clf_em = skLDA().fit(Xtr_em, ytr)
    em_err = np.mean(clf_em.predict(Xtest).flatten() != ytest)
    em_time = time.time()-start 
 
    return em_err, em_time

"""## Import MNIST"""

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

# set random seed and shuffle the data
np.random.seed(1)
idx = np.arange(len(ytrain))
np.random.shuffle(idx)
Xtrain, ytrain = Xtrain[idx,:], ytrain[idx]  

Xtrain.shape, ytrain.shape

# convert the test set to NumPy arrays and flatten the data
Xtest, ytest = [], []
for example in tfds.as_numpy(test_dataset):
  Xtest.append(example['image'].flatten())
  ytest.append(example['label'])

Xtest, ytest = np.asarray(Xtest), np.asarray(ytest)
Xtest = Xtest.astype(float)

# check if a column is all 0
id = [np.sum(Xtrain[:,i] != 0)>10 for i in range(28**2)]
# number of columns that mostly zero
print(28**2-np.sum(id))
# number of columns that has at least more than 10 non-zero
np.sum(id)

Xtrain, Xtest = Xtrain[:,id], Xtest[:,id]

Xtrain.shape, Xtest.shape

# number of sample per class in training data
ng = np.asarray([sum(ytrain==i) for i in np.arange(10)])

"""## 20%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4500,4200,3700, 3500],
                                 10).reshape((10,-1))))
p = np.array([300,320,400, 500,649])   
missing_rate(Xtrain, ytrain, n, p, 10)

em20 = compute_err_EM(Xtrain, ytrain, Xtest, ytest, n, p, 10)
print("em 20")
print(em20)

"""## 30%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4500,4200,3700, 3400],
                                 10).reshape((10,-1))))
p = np.array([100,250,350, 400,649])   
missing_rate(Xtrain, ytrain, n, p, 10)

em30 = compute_err_EM(Xtrain, ytrain, Xtest, ytest, n, p, 10)
print("em 30")
print(em30)
"""## 40%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([4000,3500,3000, 2600],
                                 10).reshape((10,-1))))
p = np.array([100,250,350, 400,649])   
missing_rate(Xtrain, ytrain, n, p, 10)

em40 = compute_err_EM(Xtrain, ytrain, Xtest, ytest, n, p, 10) 
print("em 40")
print(em40)
"""## 50%"""

n = np.hstack((ng.reshape((-1,1)), np.tile([2900,2800,2600, 2500],
                                 10).reshape((10,-1))))
p = np.array([90,140,155, 170,649])   
missing_rate(Xtrain, ytrain, n, p, 10)
em50 = compute_err_EM(Xtrain, ytrain, Xtest, ytest, n, p, 10) 
print("em 50")
print(em50)

result = np.vstack((em20, em30, em40, em50))
np.savetxt("lda_mnist_em.csv", result, delimiter=",")