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

f=f.fillna(0)
#f = f.drop(['Label'], axis = 1)
y = label['Label2'] 

#========== Training for Labeling Data -> M1 

fiturtrain, fiturtest, spam_nospam_train, spam_nospam_test = train_test_split(f ,y, test_size=0.2, stratify=y, random_state=20)
    
from sklearn import svm
svm_model= svm.SVC(kernel='linear')
svm_model.fit(fiturtrain, spam_nospam_train)
svm_y_pred_train=svm_model.predict(fiturtest)

print("Ini yang prediksi train test berlabel")

from sklearn import metrics
csvmtrain = metrics.confusion_matrix(spam_nospam_test, svm_y_pred_train)
print(csvmtrain)
    
from sklearn.metrics import classification_report
y_true_svm = spam_nospam_test
y_pred_svm = svm_y_pred_train
print(classification_report(y_true_svm, y_pred_svm))