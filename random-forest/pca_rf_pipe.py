# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:42:34 2018

@author: jtotten
"""

#### PCA and RandomForest pipeline


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set

# modeling routines from Scikit Learn packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt # for root mean-squared error calculation
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA

#import cProfile
#cProfile.run('run()')

os.chdir('C:/Users/JTOTTEN/Desktop/Data/MNIST_data')

X = pd.read_csv('C:/Users/JTOTTEN/Desktop/Data/MNIST_data/mnist_X.csv')
y = pd.read_csv('C:/Users/JTOTTEN/Desktop/Data/MNIST_data/mnist_y.csv')

#X.shape
#y.shape


# train test split
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# convert y_train to an array
conv_arr = y_train.values
y_train2 = conv_arr.ravel()

pca = PCA().fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d

from sklearn import ensemble

# create a pipeline variable for PCA -> RF
pipe = Pipeline([('pca', PCA(n_components=154)),
                 ('rf', RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, n_jobs=-1,
                                               max_features='sqrt', bootstrap=True, 
                                               class_weight="balanced_subsample"))])

# fit pipeline to data 
pipe.fit(X_train,y_train2)

# evaluate pipe on the holdout set
test_pred = pipe.predict(X_test)

# produce F1 score
jt = f1_score(y_test, test_pred, average='weighted')
jt


###############################################################################
# time

from functools import wraps

def simple_time_tracker(log_fun):
    def _simple_time_tracker(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time() - start_time

                # log the result
                log_fun({
                    'function_name': fn.__name__,
                    'total_time': elapsed_time,
                })
                
            return result

        return wrapped_fn
    return _simple_time_tracker


def _log(message):
    print('[SimpleTimeTracker] {function_name} {total_time:.3f}'.format(**message))

@simple_time_tracker(_log)
def find_components(pca):
    pca = PCA().fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(d)

@simple_time_tracker(_log)
def fitPCA_RF_pipe(X_train,y_train,X_test,y_test):
    pipe = Pipeline([('pca', PCA(n_components=154)),
                 ('rf', RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, n_jobs=-1,
                                               max_features='sqrt', bootstrap=True, 
                                               class_weight="balanced_subsample"))])
    pipe.fit(X_train,y_train)
    eval_test = pipe.predict(X_test)
    jt = f1_score(y_test, eval_test, average='weighted')
    print('Pipeline F1 Score {}'.format(jt))

@simple_time_tracker(_log)
def compress_decompress_RF_fit(X_train,y_train,X_test,y_test):
    pca = PCA(n_components=154)
    X_reduced = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)
    rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, n_jobs=-1,
                                 max_features='sqrt', bootstrap=True, class_weight="balanced_subsample")
    rnd_clf.fit(X_recovered, y_train)
    y_pred_decompression = rnd_clf.predict(X_test)
    decompress_f1 = f1_score(y_test, y_pred_decompression, average='weighted')
    print('PCA Compress - Decompressed Randfom Forest F1 Score {}'.format(decompress_f1))

@simple_time_tracker
def IncrementalPCA_RF_fit(X_train, y_train, X_test, y_test):
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)
    
    X_reduced = inc_pca.transform(X_train)
    
    rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, n_jobs=-1,
                                 max_features='sqrt', bootstrap=True, class_weight="balanced_subsample")
    
    rnd_clf.fit(X_reduced, y_train)
    incremental_y_pred = rnd_clf.predict(X_test)
    incremental_PCA_F1 = f1_score(y_test, incremental_y_pred, average='weighted')
    return incremental_PCA_F1
    
    
decompress = compress_decompress_RF_fit(X_train,y_train2,X_test,y_test)
pipe = fitPCA_RF_pipe(X_train,y_train2,X_test,y_test)
#IncrementalPCA_RF_fit(X_train, y_train2, X_test, y_test)
X_Full_Comp = find_components(X)
X_Full_Comp

model_names = ['Random Forest','PCA & RF Pipe','Decompressed RF']
times = ['60.89', '194.49', '251.21']
F1 = ['0.84',round(pipe,2),round(decompress,2)]

Model_Compare = pd.DataFrame(columns={'Model': model_names,'Execution Time (seconds)': time, 'F1 Score': F1})

from collections import OrderedDict
data = OrderedDict([('Model',['Random Forest','PCA & RF Pipe','Decompressed RF']),
                    ('Execution Time (seconds)', [60.87, 194.49, 382.55]),
                    ('F1 Score', [0.84,round(pipe,2),round(decompress,2)])])


###############################################################################
# reconstruction error
from numpy.testing import assert_array_almost_equal

# 1. estimate the components
#Xtrain = np.random.randn(100, 50)

pca_for_reconstruct = PCA(n_components=154)
pca_for_reconstruct.fit(X_train)

U, S, VT = np.linalg.svd(X_train - X_train.mean(0))

assert_array_almost_equal(VT[:154], pca_for_reconstruct.components_)

# 2. calculate he loadings

X_train_pca = pca_for_reconstruct.transform(X_train)

X_train_pca2 = (X_train - pca_for_reconstruct.mean_).dot(pca_for_reconstruct.components_.T)

assert_array_almost_equal(X_train_pca, X_train_pca2)

# 3. calculate the projection onto components in signal space 

X_projected = pca_for_reconstruct.inverse_transform(X_train_pca)
X_projected2 = X_train_pca.dot(pca_for_reconstruct.components_) + pca_for_reconstruct.mean_

assert_array_almost_equal(X_projected, X_projected2)




###############################################################################
# precision recall

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


