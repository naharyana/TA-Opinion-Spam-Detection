# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:14:52 2019

@author: hp
"""

import time
start_time = time.time()
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

test_data = pd.read_csv('Data/NonLabel_2_pf.csv', sep=',', encoding='latin-1')

#review_data = pd. read_csv('Data/......csv'), sep=',', encoding = 'latin-1'

test_data=test_data.drop(['Unnamed: 0'], axis = 1)

test_features = pd.DataFrame(columns = ['F1','F2','F3','F4','F5','F6','F7','F8','F9'
                                   ,'F10','F11','F12','F13','F14','F15','F16','F17'
                                   ,'F18','F19','F20','F21'])

test_data = test_data.rename(columns = {'Title_x':'Title_Review','Title_y':'Title_Product'})

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

#21 Review Centric Features 
   
#F1 : Jumlah feedback 
test_features['F1'] = test_data['Feedbacks']

#F2 : Jumlah helpful feedback
test_features['F2'] = test_data['HelpfulFeedbacks']

#F3 : Persentasi helpful feedback 
test_features['F3'] = test_data['HelpfulFeedbacks']/test_data['Feedbacks']

#F4 : Panjang judul review 
test_features['F4'] = test_data['Title_Review'].apply(len)

#F5 : Panjang body review
test_data['Body'].fillna('none',inplace=True)  
test_features['F5'] = test_data['Body'].apply(len)

#F6 : Posisi urutan review terlama sampai terbaru (berdasarkan tanggal) 
f6 = pd.read_csv('Data/F6.csv', sep=',', encoding='latin-1')

F6=[1 for i in range (len(f6))]
F6[0]=1

for j in range (len(f6)) :
    if j!=0:
        if f6['ProductID'][j]==f6['ProductID'][j-1]:
            F6[j]+=F6[j-1]
    
m=0        
for reviewid in test_data.loc[:,'ReviewID']:
    for j in range (len(f6)): 
        if f6['ReviewID'][j]==reviewid: 
            test_features['F6'][m]=F6[j]
    m+=1

#F7 : Posisi urutan review terbaru sampai terlama (berdasarkan tanggal)  
f7 = pd.read_csv('Data/F7.csv', sep=',', encoding='latin-1')
F7=[1 for i in range (len(f7))]
F7[0]=1

for j in range (len(f7)) :
    if j!=0:
        if f7['ProductID'][j]==f7['ProductID'][j-1]:
            F7[j]+=F7[j-1]
  
n=0        
for reviewid in test_data.loc[:,'ReviewID']:
    for j in range (len(f7)): 
        if f7['ReviewID'][j]==reviewid: 
            test_features['F7'][n]=F7[j]
    n+=1
    

#F8 : Biner apakah sebuah review itu adalah review pertama atau tidak 
j=0
for order in test_data.loc[:,'ReviewOrder']:
    if (order == 1):
        test_features['F8'][j]= 1
    else:
        test_features['F8'][j] = 0
    j+=1

#F9 : Biner apakah sebuah review itu adalah review satu-satunya
f9 = pd.read_csv('Data/F9, F18, F20, F21.csv', sep=',', encoding='latin-1')

F9=[0 for i in range (len(f9))]

for j in range (len(f9)) :
    if f9['ReviewOrder'][j]==1&f9['ReviewOrder'][j]==f9['ReviewOrder'][j+1]:
        F9[j]+=1
n=0        
for reviewid in test_data.loc[:,'ReviewID']:
    for j in range (len(f9)): 
        if f9['ReviewID'][j]==reviewid: 
            test_features['F9'][n]=F9[j]
    n+=1
    
#F10 : Persentasi kata-kata opini positif dalam review 
negative_word = pd.read_csv('Data/negative-word.csv', sep=',', encoding='latin-1')   
positive_word = pd.read_csv('Data/positive-word.csv', sep=',',encoding='latin-1')

y=len(test_data)
a=[0 for data in range(y)]
k=0
for review in test_data.loc[:,'Body']:
    #if (k==1):
    for j in range (len(positive_word)):
        if positive_word['word'][j] in review:
            a[k]+=1
            #print (positive_word['word'][j])   
    k+=1

test_features['F10'] = a

#F11 : Persentasi kata-kata negative dalam review 
b=[0 for i in range(y)]
k=0
for review in test_data.loc[:,'Body']:
    #if (k==1):
    for j in range (len(negative_word)):
        if negative_word['word'][j] in review:
            b[k]+=1
            #print (positive_word['word'][j])    
    k+=1

test_features['F11'] = b

test_data.to_csv('test_data.csv',index=False)

#F13 : Jumlah nama brand yang disebut.
c=[0 for i in range(y)]
j=0
k=0
for review in test_data.loc[:,'Body']:
    for j in range (y):
        if test_data['Brand'][j] in review:
            c[k]+=1
            #print (test_data['Brand'][j] in i)
            #print (test_data['Brand'][j])    
    k+=1

test_features['F13'] = c

#F14 : Persentase numeral dalam review.
d=[0 for i in range(y)]
j=0
for review in test_data.loc[:,'Body']:
    for letter in review: 
        if (letter.isnumeric()) == True: 
            d[j]+=1
    j+=1

test_features['F14'] = d/test_features['F5']

#F15 : Persentase huruf kapital dalam review. 
j=0
e=[0 for i in range(y)]
for review in test_data.loc[:,'Body']:
    e[j] = sum(1 for char in review if char.isupper())
    j+=1

test_features['F15'] = e/test_features['F5']

#F16 : Persentase kata kata ber huruf semua kapital dalam review.
j=0
f=[0 for i in range(y)]
for review in test_data.loc[:,'Body']:
    text = review.split()
    f[j] = sum(1 for c in text if c.isupper())
    j+=1

test_features['F16'] = f/test_features['F5']

#F17 : Rating review.  
test_features['F17'] = test_data['Rating']
    
#F18 : Deviasi rating dari rating produk.
product=1

for j in range (len(f9)-1) :
    if f9['ProductID'][j]!=f9['ProductID'][j+1]:
            product+=1

sumproductrating=[0 for i in range (product)]

count=0
nrating=[0 for i in range (product)]
pr=["" for i in range (product)]
for j in range (len(f9)) :
    if j==0:
        sumproductrating[count]+=f9['Rating'][0]
    else:
        if f9['ProductID'][j]!=f9['ProductID'][j-1]:
            count+=1
        sumproductrating[count]+=f9['Rating'][j]
    nrating[count]+=1
    pr[count]=f9['ProductID'][j]
    
ratarata = [i / j for i, j in zip(sumproductrating, nrating)] 
subrating =[0 for i in range (len(f9))]

for i in range (len(f9)):
    for j in range (len(pr)):
        if f9['ProductID'][i]==pr[j]:
            subrating[i]=abs(f9['Rating'][i]-ratarata[j])
md=[0 for i in range (product)]
i=0
for j in range (len(f9)) :
    if j==0:
        md[i]+=subrating[0]
    else:
        if f9['ProductID'][j]!=f9['ProductID'][j-1]:
            print("lala")
            md[i]=md[i]/nrating[i]
            i+=1
        #print(f9['ProductID'][j])
        md[i]+=subrating[j]

n=0            
for prodid in test_data.loc[:,'ProductID']:
    for j in range (len(pr)): 
        if pr[j]==prodid:
            test_features['F18'][n]=md[j]
    n+=1

#F19 : Kategori apakah review itu adalah bagus (rating ≥ 4), jelek (rating ≤ 2,5) atau standar (rating < 4𝑑𝑎𝑛 > 2,5)
j=0
for i in test_data.loc[:,'Rating']:
    if i>=4 :
        test_features['F19'][j]=2
    elif (i<4&i>2.5):
        test_features['F19'][j]=1
    elif i<=2.5:
        test_features['F19'][j]=0
    j+=1

#print("--- %s seconds ---" % (time.time() - start_time))

#F20 : Binary apakah review jelek ditulis setelah review bagus yang pertama dari suatu produk 
F20=[0 for i in range (len(f9))]

for j in range (len(f9)-1) :
    if ((f9['ReviewOrder'][j]==1) & (f9['ReviewOrder'][j+1]!=1) & (f9['Rating'][j]>=4) & (f9['Rating'][j+1]<=2.5)):
        F20[j+1]+=1
    
n=0            
for reviewid in test_data.loc[:,'ReviewID']:
    for j in range (len(f9)): 
        if f9['ReviewID'][j]==reviewid: 
            test_features['F20'][n]=F20[j]
    n+=1


#F21 : Binary apakah review bagus ditulis setelah review jelek yang pertama dari suatu produk
F21=[0 for i in range (len(f9))]

for j in range (len(f9)-1) :
    if ((f9['ReviewOrder'][j]==1) & (f9['ReviewOrder'][j+1]!=1) & (f9['Rating'][j]<=2.5) & (f9['Rating'][j+1]>=4)):
        F21[j+1]+=1
    
n=0            
for reviewid in test_data.loc[:,'ReviewID']:
    for j in range (len(f9)): 
        if f9['ReviewID'][j]==reviewid: 
            test_features['F21'][n]=F21[j]
    n+=1
    
print("--- %s seconds ---" % (time.time() - start_time))

standardeviasi = pd.DataFrame(columns=['pr','nrating','sumproductrating','ratarata','md'])

standardeviasi['pr']=pr
standardeviasi['nrating']=nrating
standardeviasi['sumproductrating']=sumproductrating
standardeviasi['ratarata']=ratarata
standardeviasi['md']=md

standardeviasi.to_csv('standardeviasi.csv', index=False)

fiturreview = pd.DataFrame(columns=['reviewid','f6','f7','f9','f18','f20','f21'])

fiturreview['reviewid']=f9['ReviewID']
fiturreview['F6']=F6
fiturreview['F7']=F7
fiturreview['F9']=F9
fiturreview['F18']=F20
fiturreview['F20']=F21

fiturreview.to_csv('fiturreview.csv', index=False)

test_features.to_csv('testfeatures.csv', index=False)
test_features.to_excel('testfeatures.xlsx', index=False)


#F12 : Cosine similarity dari review dan fitur produk 

start_time = time.time()
import pandas as pd
import numpy as np 

g={}
for j in range(y):
    h=[]
    h.append(test_data['Body'][j])   
    for i in test_data.loc[j, : ]:
        i=str(i)
        if i.startswith("Feature->"):
            h.append(i[9:]) 
    g[j] = h

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf_vectorizer = TfidfVectorizer()

mean_cosine=[0 for i in range(y)]
for i in range (y):
    if len(g[i])!=1:
        tfidf_fitur = tfidf_vectorizer.fit_transform(g[i])
        cos=cosine_similarity(tfidf_fitur[0:1], tfidf_fitur[1:])
        mean_cosine[i]=np.mean(cos)
    else:
        mean_cosine[i]=0

test_features['F12'] = mean_cosine

test_features.to_csv('NonLabel_2_features.csv', index=False)
test_features.to_excel('NonLabel_2_features.xlsx', index=False)

print("--- %s seconds ---" % (time.time() - start_time))
