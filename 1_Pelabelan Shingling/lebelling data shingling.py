# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:50:14 2019

@author: hp
"""
import pandas as pd

shinglingsresult=pd.read_csv('revidshingling.csv', sep=',', encoding='latin-1')
shinglingdata=pd.read_csv('Shingling_New_5719.csv', sep=',', encoding='latin-1')

shinglingspamdata=pd.DataFrame()
shinglingnotspamdata=pd.DataFrame()

data=len(shinglingdata)

for i in shinglingsresult['SpamID']:
    for j in range (data):
        if i == shinglingdata['ReviewID'][j]:
            shinglingspamdata=shinglingspamdata.append(shinglingdata.iloc[j], ignore_index='true')
            
for i in shinglingsresult['NotspamID']:
    for j in range (data):
        if i == shinglingdata['ReviewID'][j]:
            shinglingnotspamdata=shinglingnotspamdata.append(shinglingdata.iloc[j], ignore_index='true')
            
shinglingspamdata['Label'] = '1'
shinglingnotspamdata['Label'] = '0'

shingling=pd.concat([shinglingspamdata, shinglingnotspamdata], ignore_index=True, sort=False)

shingling.to_csv('shingling.csv', index=False)


shinglingspamdata.to_csv('shingling_spam.csv', index=False)

shinglingnotspamdata.to_csv('shingling_notspam.csv', index=False)