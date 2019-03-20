# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:13:56 2019

@author: hp
"""

import pandas as pd

df = pd.read_csv('productInfoXML-reviewed-mProductssv.csv', sep=',', encoding='latin-1')
df1 = pd.read_csv('labeledSpam.csv', sep=',', encoding='latin-1')

#df3= df[:50000]

jumlah_data = len(df)
jumlah_data_spam = len (df1)
df2 = df1
j=79800

#print(df[1:3]) data dari index ke 1 sampai 3

#print (df1['ProductID'][1])
#df2=df1.append(df.loc[4, :], sort = False)
#print(df.loc[4, : ])
#print(df.loc[4:5 ])
#df3 = pd.concat([df1,df], axis =1) kalau yang ini ngikut yang df hmmmmm

#cols = [c for c in df.columns if c.lower()[:4] != 'test']

#df.drop(df.columns[df.columns.str.contains('Feature->',case = False)],axis = 1)
#j=79800

#print(df[1:3]) data dari index ke 1 sampai 3

#print (df1['ProductID'][1])

#print(df.loc[4, : ]) data index 4

#df2 = df1[0:3] #kalau mau akses data ke 0 sampe 2 
#for i in range (jumlah_data):
 #       if df['Unnamed: 62'].str.contains('Feature->'):
  #          print ("tes")

#udah bisa nyamain tapi line 2 yang ada fiturnya masih belum ngikut

