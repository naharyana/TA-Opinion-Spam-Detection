# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:42:30 2019

@author: Ana Alimatus Zaqiyah
"""

import pandas as pd

#KETERANGAN DATAFRAME :

#df = data produk
#df1 = data review terlabel
#df2 = temp data fitur 
#df3 = tampat kopi data produk yang kemudia tidak ada fiturnya
#df4 = tempat pengecekan apakah ada kolom yang ada fiturnya atau tidak
#df5 = tempat hasil penggabungan data asli produk dan fiturnya

#df 6 = tempat oengecekan apakah kolum nya ada fitur nya apa nggak setelah kolum kelima [2]


df = pd.read_csv('productInfoXML-reviewed-mProductssv.csv', sep=',', encoding='latin-1')

df1 = pd.read_csv('labeledSpam.csv', sep=',', encoding='latin-1')
df2 = pd.DataFrame()
jumlah_data = len(df)

#hapus kolom yang tidak ada featurenya
count = 0  
for column in df:
    count+=1
    if count>5:
        df4= df[df[column].str.contains('Feature->', na=False)]
        if df4.empty:
            df=df.drop([column], axis = 1)
            
#drop break reviewed           
df = df.dropna(subset=['Sales Rank'])
    
#reset index
df = df.reset_index(drop=True)

#data produk
df3 = df

#pemisahan baris data asli produk dan fitur

#data fitur
for count in range (len(df3)) :
   print(count)
   if count%2==1 :
       df2 = df2.append(df3.loc[count, : ] ) 

#drop yang fitur
for count in range (len(df3)) :
   if count%2==1 :
       df3=df3.drop(count)

#reset index untuk memudahkan concenate
df2 = df2.reset_index(drop=True)
df3 = df3.reset_index(drop=True)

#penggantian nama header agar tidak rancu saat diconcenate
df2.columns = range(df2.shape[1])

#masukin ke excel

df2.to_excel("df2.xlsx")

#drop column ga kepake

df3 = df3.dropna(axis=1, how ='all')

df3.to_excel("df3.xlsx")

#penggabungan baris data asli dan data fitur

df5=pd.concat([df3,df2], axis=1, sort = False)

df5.to_excel("df5.xlsx")

#pengecekan [2] untuk kolom yang non feature->

count = 0  
for column in df5:
    count+=1
    if count>5:
        df6= df5[df5[column].str.contains('Feature->', na=False)] #yang berisi feature
        if df6.empty:
            df5=df5.drop([column], axis = 1)

#df5.to_excel("df5-2.xlsx") ga perlui

#merge info produk dan review berdasarkan kesamaan product id
#df7 = pd.merge(df1, df5 , on=['ProductID']) 

#df7.to_excel("df5.xlsx")

#jump to df5 read