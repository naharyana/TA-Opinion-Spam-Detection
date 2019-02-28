# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:42:30 2019

@author: hp
"""

import pandas as pd

df = pd.read_csv('productInfoXML-reviewed-mProductssv.csv', sep=',', encoding='latin-1')
df1 = pd.read_csv('labeledSpam.csv', sep=',', encoding='latin-1')

#df3= df[:50000]

#jumlah_data = len(df)
jumlah_data_spam = len (df1)

print(df.loc[4, : ])

print  (df1[1][2])
df2 = df1[0:3] #kalau mau akses data ke 0 sampe 2 
for i in range (jumlah_data):
    for j in range (jumlah_data_spam):
        if df1['ProductID'][j]== df['<ProductID>'][i]:
            print(df1.loc[df1['ProductID'][j], : ])
            print (df1['ProductID'][j])
            df1.append()
            
    
    #print (df1['ReviewID'][0])