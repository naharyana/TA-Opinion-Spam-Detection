# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:42:30 2019

@author: hp
"""

import pandas as pd

#df = pd.read_csv('productInfoXML-reviewed-mProductssv.csv', sep=',', encoding='latin-1')
df1 = pd.read_csv('labeledSpam.csv', sep=',', encoding='latin-1')

jumlah_data = len(df)
jumlah_data_spam = len (df1)

#df2 = df1[0:3] #kalau mau akses data ke 0 sampe 2 
#for in range jumlah_data:
    