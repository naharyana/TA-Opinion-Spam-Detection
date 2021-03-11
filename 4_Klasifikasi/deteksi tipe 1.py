# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:54:04 2019

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:01:28 2019

@author: hp
"""

import time
start_time = time.time()
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizertfidf = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

spamshingling = pd.read_csv('Data/shinglingspamfeatures.csv', sep=',', encoding='latin-1')
notspamshingling = pd.read_csv('Data/shinglingnotspamfeatures.csv', sep=',', encoding='latin-1')

trains_features = pd.read_csv('shingling_spam_pf.csv', sep=',', encoding='latin-1')
trainns_features = pd.read_csv('shingling_notspam_pf.csv', sep=',', encoding='latin-1')

train_features =pd.concat([spamshingling, notspamshingling], ignore_index=True, sort=False)
sample =pd.concat([trains_features, trainns_features], ignore_index=True, sort=False)

corpus=sample['Body']

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

corpus1=corpus.apply(text_preprocess)

y = sample['Label'] 

#merge corpus test and train

X = vectorizertfidf.fit_transform(corpus1)

ngram = pd.DataFrame(X.toarray())
train_features=train_features.fillna(0)

allfitur=ngram

#split train tes TRAIN

message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(allfitur ,y, test_size=0.3, stratify=y, random_state=20)
    
from sklearn import svm
svm_model= svm.SVC(kernel='linear')
svm_model.fit(message_train, spam_nospam_train)
svm_y_pred_train=svm_model.predict(message_test)

print("Klasifikasi TIpe Spam 1")

from sklearn import metrics
csvmtrain = metrics.confusion_matrix(spam_nospam_test, svm_y_pred_train)
print(csvmtrain)

from sklearn.metrics import classification_report
y_true_svm = spam_nospam_test
y_pred_svm = svm_y_pred_train
print(classification_report(y_true_svm, y_pred_svm))

print("--- %s seconds ---" % (time.time() - start_time))