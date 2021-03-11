# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:53:46 2019

@author: hp
"""
#IMPORT LIBRARY
import time
start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#READ DATA REVIEW
sample = pd.read_csv('Data/train_data.csv', sep=',', encoding='latin-1')

#READ DATA FITUR
f = pd.read_csv('Data/ngram-nostemming.csv', sep=',', encoding='latin-1')
label = pd.read_csv('label01.csv', sep=',', encoding='latin-1')
lis=[]
for i in range(len(f)):
    lis.append(f.iloc[i])

#f=f.fillna(0)
#f = f.drop(['Label'], axis = 1)
y_l = label['Label2'] 

#========== Training for Labeling Data -> M1 

import pickle
from sklearn import svm
#stratified kfold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report
import sklearn.metrics as sm
n_fold = 10
skf = StratifiedKFold(n_splits = n_fold )
clf = svm.SVC(kernel='linear', C=5, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=2, gamma='auto', max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svm_model= svm.SVC(kernel='linear')
svm_accuracy = 0
i=0
for train_index, test_index in skf.split(lis, y_l):
    #train
    attribute = []
    kelas = []
    for x in train_index:
        attribute.append(lis[x])
        kelas.append(y_l[x])
    x_train = pd.DataFrame(attribute)
    y_train = pd.DataFrame(kelas)
    attribute = []
    kelas = []
    #test
    for l in test_index:
        attribute.append(lis[l])
        kelas.append(y_l[l])
    x_test = pd.DataFrame(attribute)
    y_test = pd.DataFrame(kelas)
    svc = clf.fit(x_train, y_train)
    filename = str(i)+'kontri.pickle'
    pickle.dump(clf, open(filename, 'wb'))
    svm_result = svc.predict(x_test)
    
    print("Ini yang prediksi train test berlabel")
    
    csvmtrain = metrics.confusion_matrix(y_test, svm_result)
    print(csvmtrain)
    
    y_true_svm = y_test
    y_pred_svm = svm_result
    print(classification_report(y_true_svm, y_pred_svm))

    akurasi = float(sm.accuracy_score(svm_result, y_test)) * 100
    print(akurasi)
    svm_accuracy += akurasi
print('Akurasi Support Vector Machine : ' + repr(svm_accuracy / n_fold))