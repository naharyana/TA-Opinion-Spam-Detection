# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:01:35 2020

@author: hp
"""

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
sample = pd.read_csv('train_data - Copy.csv', sep=',', encoding='latin-1')

#READ DATA FITUR
#f = pd.read_csv('Data/ngram-nostemming.csv', sep=',', encoding='latin-1')
#label = pd.read_csv('label01.csv', sep=',', encoding='latin-1')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

vectorizertfidf1 = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

import string
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

corpus=sample['Body'].apply(text_preprocess)

A = vectorizertfidf1.fit_transform(corpus)
ngram = pd.DataFrame(A.toarray())

#f = f.drop(['Label'], axis = 1)
y = sample['Label'] 

#========== Training for Labeling Data -> M1 

fiturtrain, fiturtest, spam_nospam_train, spam_nospam_test = train_test_split(ngram ,y, test_size=0.3, stratify=y, random_state=20)
    
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