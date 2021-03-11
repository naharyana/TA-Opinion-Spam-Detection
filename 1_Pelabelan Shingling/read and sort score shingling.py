# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:50:55 2019

@author: hp
"""

import pandas as pd

shinglingscore=pd.read_csv('Shingling_New_5719_Score.csv', sep=',', encoding='latin-1')

#shinglingsort=shinglingscore.sort_values(by=['score'], ascending=False)
shinglingsort=shinglingscore.sort_values(by=['score'])

shinglingsorten=shinglingsort.head(n=200)

shinglingsorten.to_csv('shinglingsortpart.csv', index=False)