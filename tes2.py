# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:35:30 2019

@author: hp
"""

import pandas as pd

import numpy as np

array=np.random.random((2,4))

df = pd.DataFrame({'name': ['Bob', 'gfdgd', 'Alice','dsadsa','dsewds','io'], 
                   'brand': ['tend', 'football', 'basketball','we','ddgfsds','lo'],
                   'title': ['tennis', 'feature->', 'bassket','feature->','dsfdsfds','pl'],
                   'tFDle': ['tennis', 'feature->', 'bassket','feature->','dsfdsfds','pl'],
                   'le': ['tennis', 'feature->', 'bassket','feature->','dsfdsfds','pl'],
                   'un1': ['', 'lalala', '','re','','ds']}, index=[0, 1, 2, 3, 4, 5])


print (df)

df4=pd.DataFrame()
x=len(df)
print(x)
#for row in df:
for count in range (len(df)) :
   print(count)
   if count%2==1 :
       df4 = df4.append(df.loc[count, : ] ) 

for count in range (len(df)) :
   if count%2==1 :
       df=df.drop(count)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

df4.columns = range(df4.shape[1])

df2=pd.concat([df,df4], axis=1, sort = False)

df7= df2[df2['brand'].str.contains('feature->', na=True)] #yang berisi feature

#print (df2)

