# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:15:28 2019

@author: hp
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import string
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

@app.route('/')
def home():
	return render_template('beranda.html')

@app.route('/testing')
def spam():
    df= pd.read_csv("Data/datareview.csv", encoding="latin-1")
    review = df['Body']
    judul = df['Title_Product']
    return render_template('spam.html', body=review, title=judul)

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("Data/datareview.csv", encoding="latin-1")
    #MODEL BIGRAM DETEKSI 1
    trains_features = pd.read_csv('Data/shingling_spam_pf.csv', sep=',', encoding='latin-1')
    trainns_features = pd.read_csv('Data/shingling_notspam_pf.csv', sep=',', encoding='latin-1')
    sample =pd.concat([trains_features, trainns_features], ignore_index=True, sort=False)
    corpus=sample['Body']
    corpus1=corpus.apply(text_preprocess)
    y = df['Label'] 
    vectorizertfidf = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)
    X = vectorizertfidf.fit_transform(corpus1)
    
    #DETEKSI23 
    fiturshing=pd.read_csv('Data/shingfeatures.csv', sep=',', encoding='latin-1')
    sample0 = pd.read_csv('Data/train_data.csv', sep=',', encoding='latin-1')
    sample1 = pd.read_csv('Data/NonLabel_1_pf.csv', sep=',', encoding='latin-1')
    sample2 = pd.read_csv('Data/NonLabel_2_pf.csv', sep=',', encoding='latin-1')
    sample3 = pd.read_csv('Data/NonLabel_3_pf.csv', sep=',', encoding='latin-1')
    sample4 = pd.read_csv('Data/NonLabel_4_pf.csv', sep=',', encoding='latin-1')
    corpus0=sample0['Body']
    corpus1=sample1['Body']
    corpus2=sample2['Body']
    corpus3=sample3['Body']
    corpus4=sample4['Body']
    corpus0=corpus0.apply(text_preprocess)
    corpus1=corpus1.apply(text_preprocess)
    corpus2=corpus2.apply(text_preprocess)
    corpus3=corpus3.apply(text_preprocess)
    corpus4=corpus4.apply(text_preprocess)
    corpusall = pd.concat([corpus0, corpus1, corpus2, corpus3, corpus4], keys=['lb', 'ul1', 'ul2', 'ul3', 'ul4'])
    vectorizertfidf1 = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)
    Y = vectorizertfidf1.fit_transform(corpusall)

    kombinasi23 = pd.read_csv('fitur-nostem-t.csv', sep=',', encoding='latin-1')

    filename = 'shingling-gram-model93-linear.pickle'
    filename2 = 'nostem-bd1-model92-poly-tfidf.pickle'
    loaded_model = pickle.load(open(filename, 'rb')) 
    loaded_model2 = pickle.load(open(filename2, 'rb')) 
    
    if request.method == 'POST':
        idrev = request.form['indexreview'] 
        data = int(idrev)
        review=df['Body'][data]
        review=[review]
        
        tst=vectorizertfidf.transform(review)
        testing = pd.DataFrame(tst.toarray())
        
        if (data<12):
            testing2=vectorizertfidf1.transform(review)
            testing2 = pd.DataFrame(testing2.toarray()) 
            fiturs=np.array(fiturshing.loc[data,:])
            fiturs=np.reshape(fiturs, (1, -1))
            fiturs=pd.DataFrame(testing, columns=fiturshing.columns)
            fitur=pd.concat([fiturs, testing2], axis=1, join_axes=[fiturs.index])
            fitur=fitur.fillna(0)
        else: #23
            fiturs=np.array(kombinasi23.loc[data-12,:])
            fitur=np.reshape(fiturs, (1, -1))
         
        y_pred=loaded_model.predict(testing)
        if y_pred==0:
            y_pred=loaded_model2.predict(fitur)
        true=y[data]
        true=int(true)
        judul = df['Title_Product'][data]
        review =df['Body'][data]
    return render_template('result.html',prediction= y_pred, y_true=true, review=review, judul=judul)


if __name__ == '__main__':
	app.run(debug=True)
   