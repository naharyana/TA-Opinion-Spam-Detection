from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import string
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizertfidf = TfidfVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b', min_df=1)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('beranda.html')

@app.route('/testing23')
def spam23():
    df= pd.read_csv("train_data.csv", encoding="latin-1")
    review = df['Body']
    judul = df['Title_Product']
    return render_template('spam23.html', body=review, title=judul)

@app.route('/predict23',methods=['POST'])
def predict23():
    #READ DATA FITUR
    sample = pd.read_csv('train_data.csv', sep=',', encoding='latin-1')
    f = pd.read_csv('fitur-nostem-t.csv', sep=',', encoding='latin-1')
    y = sample['Label']
    #f = f.drop(['Label'], axis = 1)
    fitur=f         
    filename = 'nostem-bd1-model92-poly-tfidf.pickle'
    loaded_model = pickle.load(open(filename, 'rb')) 
    if request.method == 'POST':
        idrev = request.form['indexreview']
        data = int(idrev)
        testing=fitur.loc[data,:]
        testing=np.array(fitur.loc[data,:])
        testing=np.reshape(testing, (1, -1))
        y_prediction=loaded_model.predict(testing)
        true=y[data]
        true=int(true)
        review=sample['Body'][data]
        judul = sample['Title_Product'][data]
    return render_template('result23.html',prediction = y_prediction, y_true=true, review=review, title=judul)

@app.route('/testing1')
def spam1():
    df= pd.read_csv("shingling_pf.csv", encoding="latin-1") 
    sample= pd.read_csv("shingling_pf.csv", encoding="latin-1")
    f = pd.read_csv('bigram-t-nostem-s.csv', sep=',', encoding='latin-1')
    fitur=f         
    y = sample['Label']
    filename = 'shingling-gram-model93-linear.pickle'
    loaded_model = pickle.load(open(filename, 'rb')) 
    
    if request.method == 'POST':
        idrev = request.form['indexreview1']
        data = int(idrev)
        testing=fitur.loc[data,:]
        testing=np.array(fitur.loc[data,:])
        testing=np.reshape(testing, (1, -1))
        corpus=sample['Body'][data]
        y_prediction=loaded_model.predict(testing)
        true=y[data]
        true=int(true)
        judul = sample['Title_y'][data]
    return render_template('result1.html',prediction1= y_prediction, y_true1=true, review1=corpus, title1=judul)

if __name__ == '__main__':
	app.run(debug=True)