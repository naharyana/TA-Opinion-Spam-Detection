# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:34:39 2019

@author: hp
"""

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#vectorizer = CountVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

vectorizertfidf1 = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

sample = pd.read_csv('Data/train_data.csv', sep=',', encoding='latin-1')
sample1 = pd.read_csv('Data/NonLabel_1_pf.csv', sep=',', encoding='latin-1')
sample2 = pd.read_csv('Data/NonLabel_2_pf.csv', sep=',', encoding='latin-1')
sample3 = pd.read_csv('Data/NonLabel_3_pf.csv', sep=',', encoding='latin-1')
sample4 = pd.read_csv('Data/NonLabel_4_pf.csv', sep=',', encoding='latin-1')

#trains_features = pd.read_csv('shingling_spam_pf.csv', sep=',', encoding='latin-1')
#trainns_features = pd.read_csv('shingling_notspam_pf.csv', sep=',', encoding='latin-1')
#samples =pd.concat([trains_features, trainns_features], ignore_index=True, sort=False)

#spamshingling = pd.read_csv('Data/shinglingspamfeatures.csv', sep=',', encoding='latin-1')
#notspamshingling = pd.read_csv('Data/shinglingnotspamfeatures.csv', sep=',', encoding='latin-1')
#train_features =pd.concat([spamshingling, notspamshingling], ignore_index=True, sort=False)

#igram23 = pd.read_csv('Data/kombinasi-shingling-t-nostem.csv', sep=',', encoding='latin-1')
#bigram231 = pd.read_csv('Data/bigram-t-nostem.csv', sep=',', encoding='latin-1')

f = pd.read_csv('Data/trainfeatures.csv', sep=',', encoding='latin-1')
f1 =pd.read_csv('Data/NonLabel_1_features.csv', sep=',', encoding='latin-1')
f2 =pd.read_csv('Data/NonLabel_2_features.csv', sep=',', encoding='latin-1')
f3 =pd.read_csv('Data/NonLabel_3_features.csv', sep=',', encoding='latin-1')
f4 =pd.read_csv('Data/NonLabel_4_features.csv', sep=',', encoding='latin-1')

corpus=sample['Body']
corpus1=sample1['Body']
corpus2=sample2['Body']
corpus3=sample3['Body']
corpus4=sample4['Body']

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
corpus=sample['Body']
corpus1=sample1['Body']
corpus2=sample2['Body']
corpus3=sample3['Body']
corpus4=sample4['Body']
corpus=corpus.apply(text_preprocess)
#corpus=corpus.apply(stemmer)

corpus1=corpus1.apply(text_preprocess)
#corpus1=corpus1.apply(stemmer)

corpus2=corpus2.apply(text_preprocess)
#corpus2=corpus2.apply(stemmer)

corpus3=corpus3.apply(text_preprocess)
#corpus3=corpus3.apply(stemmer)

corpus4=corpus4.apply(text_preprocess)
#corpus4=corpus4.apply(stemmer)

corpusall = pd.concat([corpus, corpus1, corpus2, corpus3, corpus4], keys=['lb', 'ul1', 'ul2', 'ul3', 'ul4'])

#X = vectorizer.fit_transform(corpusall)
X = vectorizertfidf1.fit_transform(corpusall)
ngram = pd.DataFrame(X.toarray())

lb=corpusall.loc['lb']
ul1=corpusall.loc['ul1']
ul2=corpusall.loc['ul2']
ul3=corpusall.loc['ul3']
ul4=corpusall.loc['ul4']

a=len(lb)-1
b=len(ul1)
c=len(ul2)
d=len(ul3)
e=len(ul4)

testngram=ngram.loc[0:a]
testngram1=ngram.loc[a+1:a+b]  
testngram2=ngram.loc[a+b+1:a+b+c]
testngram3=ngram.loc[a+b+c+1:a+b+c+d]
testngram4=ngram.loc[a+b+c+d+1:a+b+c+d+e]

fitur=pd.concat([f, testngram], axis=1, join_axes=[f.index])
fitur1=pd.concat([f1, testngram1], axis=1, join_axes=[f1.index])
fitur2=pd.concat([f2, testngram2], axis=1, join_axes=[f2.index])
fitur3=pd.concat([f3, testngram3], axis=1, join_axes=[f3.index])
fitur4=pd.concat([f4, testngram4], axis=1, join_axes=[f4.index])

fitur=fitur.fillna(0)
fitur1=fitur1.fillna(0)
fitur2=fitur2.fillna(0)
fitur3=fitur3.fillna(0)
fitur4=fitur4.fillna(0)

fitur.to_csv('nfitur-nopre.csv', index=False)
fitur1.to_csv('nfitur1-nopre.csv', index=False)
fitur2.to_csv('nfitur2-nopre.csv', index=False)
fitur3.to_csv('nfitur3-nopre.csv', index=False)
fitur4.to_csv('nfitur4-nopre.csv', index=False)

'''
bigram :

testngram.to_csv('bigram-t-nostem.csv', index=False)
testngram1.to_csv('bigram1-t-nostem.csv', index=False)
testngram2.to_csv('bigram2-t-nostem.csv', index=False)
testngram3.to_csv('bigram3-t-nostem.csv', index=False)
testngram4.to_csv('bigram4-t-nostem.csv', index=False)
'''