# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:03:36 2019

@author: hp
"""

import pandas as pd

#Data Testing

df5 = pd.read_csv('df5.csv', sep=',', encoding='latin-1')

df5=df5.drop(['Unnamed: 0'], axis = 1)

#sesuaikan nama file dengan data training maupun testing
df1 = pd.read_csv('namafile.csv', sep=',', encoding='latin-1')

'''
training : 
    
df1 = pd.read_csv('labeledNotSpam.csv', sep=',', encoding='latin-1')
df2 = pd.read_csv('labeledSpam.csv', sep=',', encoding='latin-1')

testing :
    
df1 = pd.read_csv('NonLabel_x.csv', sep=',', encoding='latin-1')

'''

df2 = pd.merge(df1, df5 , on=['ProductID']) 

#sesuaikan nama testing atau training
df2.to_csv('NonLabel_4_pf.csv', index=False)
