# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 19:51:34 2019

@author: hp
"""
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

vectorizertfidf1 = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

sample = pd.read_csv('Data/train_data.csv', sep=',', encoding='latin-1')
sample1 = pd.read_csv('Data/NonLabel_1_pf.csv', sep=',', encoding='latin-1')
sample2 = pd.read_csv('Data/NonLabel_2_pf.csv', sep=',', encoding='latin-1')
sample3 = pd.read_csv('Data/NonLabel_3_pf.csv', sep=',', encoding='latin-1')
sample4 = pd.read_csv('Data/NonLabel_4_pf.csv', sep=',', encoding='latin-1')

f = pd.read_csv('Data/trainfeatures.csv', sep=',', encoding='latin-1')
f1 =pd.read_csv('Data/NonLabel_1_features.csv', sep=',', encoding='latin-1')
f2 =pd.read_csv('Data/NonLabel_2_features.csv', sep=',', encoding='latin-1')
f3 =pd.read_csv('Data/NonLabel_3_features.csv', sep=',', encoding='latin-1')
f4 =pd.read_csv('Data/NonLabel_4_features.csv', sep=',', encoding='latin-1')

f=f.drop(columns=['Label'])

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

corpus=sample['Body'].apply(text_preprocess)
corpus1=sample1['Body'].apply(text_preprocess)
corpus2=sample2['Body'].apply(text_preprocess)
corpus3=sample3['Body'].apply(text_preprocess)
corpus4=sample4['Body'].apply(text_preprocess)

A = vectorizertfidf1.fit_transform(corpus)
B = vectorizertfidf1.transform(corpus1)
C = vectorizertfidf1.transform(corpus2)
D = vectorizertfidf1.transform(corpus3)
E = vectorizertfidf1.transform(corpus4)

ngram = pd.DataFrame(A.toarray())
ngram1 = pd.DataFrame(B.toarray())
ngram2 = pd.DataFrame(C.toarray())
ngram3 = pd.DataFrame(D.toarray())
ngram4 = pd.DataFrame(E.toarray())

fitur=pd.concat([f, ngram], axis=1, join_axes=[f.index])
fitur1=pd.concat([f1, ngram1], axis=1, join_axes=[f1.index])
fitur2=pd.concat([f2, ngram2], axis=1, join_axes=[f2.index])
fitur3=pd.concat([f3, ngram3], axis=1, join_axes=[f3.index])
fitur4=pd.concat([f4, ngram4], axis=1, join_axes=[f4.index])

fitur=fitur.fillna(0)
fitur1=fitur1.fillna(0)
fitur2=fitur2.fillna(0)
fitur3=fitur3.fillna(0)
fitur4=fitur4.fillna(0)

fitur.to_csv('nfitur-new.csv', index=False)
fitur1.to_csv('nfitur1-new.csv', index=False)
fitur2.to_csv('nfitur2-new.csv', index=False)
fitur3.to_csv('nfitur3-new.csv', index=False)
fitur4.to_csv('nfitur4-new.csv', index=False)

'''
bigram :

testngram.to_csv('bigram-t-nostem.csv', index=False)
testngram1.to_csv('bigram1-t-nostem.csv', index=False)
testngram2.to_csv('bigram2-t-nostem.csv', index=False)
testngram3.to_csv('bigram3-t-nostem.csv', index=False)
testngram4.to_csv('bigram4-t-nostem.csv', index=False)
'''

#corpus=sample['Body']
#corpus1=sample1['Body']
#corpus2=sample2['Body']
#corpus3=sample3['Body']
#corpus4=sample4['Body']
#corpusall = pd.concat([corpus, corpus1, corpus2, corpus3, corpus4], keys=['lb', 'ul1', 'ul2', 'ul3', 'ul4'])

#trains_features = pd.read_csv('shingling_spam_pf.csv', sep=',', encoding='latin-1')
#trainns_features = pd.read_csv('shingling_notspam_pf.csv', sep=',', encoding='latin-1')
#samples =pd.concat([trains_features, trainns_features], ignore_index=True, sort=False)

#spamshingling = pd.read_csv('Data/shinglingspamfeatures.csv', sep=',', encoding='latin-1')
#notspamshingling = pd.read_csv('Data/shinglingnotspamfeatures.csv', sep=',', encoding='latin-1')
#train_features =pd.concat([spamshingling, notspamshingling], ignore_index=True, sort=False)

#igram23 = pd.read_csv('Data/kombinasi-shingling-t-nostem.csv', sep=',', encoding='latin-1')
#bigram231 = pd.read_csv('Data/bigram-t-nostem.csv', sep=',', encoding='latin-1')