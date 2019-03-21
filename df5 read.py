# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:03:36 2019

@author: hp
"""

import pandas as pd

df5 = pd.read_csv('df5.csv', sep=',', encoding='latin-1')

df5=df5.drop(['Unnamed: 0'], axis = 1)

df1 = pd.read_csv('notSpam.csv', sep=';', encoding='latin-1')

df8 = pd.merge(df1, df5 , on=['ProductID']) 

df8.to_csv('Not Spam.csv')



