# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:13:03 2019

@author: hp
"""

import pandas as pd
from sklearn.metrics import jaccard_similarity_score

sample = pd.read_csv('Shingling_New_5719.csv', sep=',', encoding='latin-1')
honey=sample.dropna()
honey=honey.reset_index(drop=True)
def jaccard_similarity(set_a, set_b):
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

jaccard_score = pd.DataFrame(columns=['reviewid1','reviewid2','score'])
#sample=sample.dropna()
X=honey['Body']
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word',ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

a=[]
c=[]
d=[]
for i in range (len(X)):
    print("..")
    for j in range (i+1, len(X)):
        b=[]
        b.append(X[i])
        b.append(X[j])
        b = vectorizer.fit_transform(b)  #print(vectorizer.get_feature_names())
        b=b.toarray()
        sm=jaccard_similarity_score(b[0], b[1])
        a.append(sm)
        c.append(honey['ReviewID'][i])
        d.append(honey['ReviewID'][j])
#a.append(sim)
        
jaccard_score['score']=a
jaccard_score['reviewid1']=c
jaccard_score['reviewid2']=d

#SAVE TO Shingling_New_5719_Score