# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:50:06 2019

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:00:44 2019

@author: hp
"""
'''
#SEMISUPERVISED LEARNING SVM DETECTING OPINI SPAM

1. DATA BERLABEL
2. DATA NON LABEL 1
3. DATA NON LABEL 2
4. DATA NON LABEL 3
5. DATA NON LABEL 4

'''
#IMPORT LIBRARY
import time
start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#READ DATA REVIEW
sample = pd.read_csv('Data/train_data.csv', sep=',', encoding='latin-1')

#READ DATA FITURS
f = pd.read_csv('Data/nfitur-new.csv', sep=',', encoding='latin-1')
f1 = pd.read_csv('Data/nfitur1-new.csv', sep=',', encoding='latin-1')
f2 = pd.read_csv('Data/nfitur2-new.csv', sep=',', encoding='latin-1')
f3 = pd.read_csv('Data/nfitur3-new.csv', sep=',', encoding='latin-1')
f4 = pd.read_csv('Data/nfitur4-new.csv', sep=',', encoding='latin-1')


f=f.fillna(0)
f1=f1.fillna(0)
f2=f2.fillna(0)
f3=f3.fillna(0) 
f4=f4.fillna(0)

#f = f.drop(['Label'], axis = 1)
y = sample['Label'] 

#OVERSAMPLING DATA BERLABEL

from imblearn.over_sampling import SMOTE 

sm = SMOTE(kind='borderline1')
X_sm, y_sm = sm.fit_sample(f, y)

#========== Training for Labeling Data -> M1 
'''
fiturtrain, fiturtest, spam_nospam_train, spam_nospam_test = train_test_split(X_sm ,y_sm, test_size=0.3, stratify=y_sm, random_state=20)
    
from sklearn import svm
svm_model= svm.SVC(kernel='linear')
svm_model.fit(fiturtrain, spam_nospam_train)
svm_y_pred_train=svm_model.predict(fiturtest)

print("Ini yang prediksi train test berlabel")

from sklearn import metrics
csvmtrain = metrics.confusion_matrix(spam_nospam_test, svm_y_pred_train)
print(csvmtrain)
    
from sklearn.metrics import classification_report
y_true_svm = spam_nospam_test
y_pred_svm = svm_y_pred_train
print(classification_report(y_true_svm, y_pred_svm))

#========== M1 Predict UNLABELED1 

svm_model.fit(X_sm, y_sm)
svm_y_pred_f1=svm_model.predict(f1)

#==========     Updated1 = Labeling Data + Predicted Unlabeled1 

Xsm=pd.DataFrame(np.array(X_sm), columns=f.columns)

updated1=pd.concat([Xsm, f1], ignore_index=True, sort=False)

updated1=updated1.fillna(0)
ypred=pd.DataFrame(np.array(y_sm))
ypredf1=pd.DataFrame(np.array(svm_y_pred_f1))

y_updated1=pd.concat([ypredf1, ypred], ignore_index=True, sort=False)

y_updated1=y_updated1.fillna(0)
#==========     Classification Updated1
print("Klasifikasi updated 1")
fiturs_train_1, fiturs_test_1, spam_nospam_u_train_1, spam_nospam_u_test_1 = train_test_split(updated1 ,y_updated1, test_size=0.3, stratify=y_updated1, random_state=20)
svm_model.fit(fiturs_train_1, spam_nospam_u_train_1)
svm_y_pred_u1=svm_model.predict(fiturs_test_1)

from sklearn import metrics
csvmsemi1 = metrics.confusion_matrix(spam_nospam_u_test_1, svm_y_pred_u1)
print(csvmsemi1)

#=pd.DataFrame(svm_y_pred)

from sklearn.metrics import classification_report
y_true_svm_1 = spam_nospam_u_test_1
y_pred_svm_1 = svm_y_pred_u1
print(classification_report(y_true_svm_1, y_pred_svm_1))


#========== Keep The True from Updated1 

y_true_svm_1 = y_true_svm_1.reset_index()

temp=pd.DataFrame()
ytemp=pd.DataFrame()

for i in range(len(y_true_svm_1)):
    if (y_true_svm_1[0][i])!=(y_pred_svm_1[i]):
        print(i)
        ytemp=ytemp.append(y_updated1.loc[i], ignore_index=True)
        temp=temp.append(updated1.loc[i])
        updated1 = updated1.drop(i)
        y_updated1 =y_updated1.drop(i)

#========== Training for Updated1 -> M3
        
svm_model.fit(updated1, y_updated1) 
        
#========== M3 Predict UNLABELED2 

svm_y_pred_f2=svm_model.predict(f2)   

#==========     Updated2 = Updated1 + Predicted UNLABELED2 

updated2=pd.concat([updated1, f2], ignore_index=True, sort=False)

ypredf2=pd.DataFrame(np.array(svm_y_pred_f2))

y_updated2=pd.concat([y_updated1, ypredf2], ignore_index=True, sort=False)

#==========     Classification Updated2
print("Klasifikasi updated 2")
fiturs_train_2, fiturs_test_2, spam_nospam_u_train_2, spam_nospam_u_test_2 = train_test_split(updated2 ,y_updated2, test_size=0.3, stratify=y_updated2, random_state=20)
svm_model.fit(fiturs_train_2, spam_nospam_u_train_2)
svm_y_pred_u2=svm_model.predict(fiturs_test_2)

from sklearn import metrics
csvmsemi2 = metrics.confusion_matrix(spam_nospam_u_test_2, svm_y_pred_u2)
print(csvmsemi2)

#=pd.DataFrame(svm_y_pred)

from sklearn.metrics import classification_report
y_true_svm_2 = spam_nospam_u_test_2
y_pred_svm_2 = svm_y_pred_u2
print(classification_report(y_true_svm_2, y_pred_svm_2))

#========== Keep The True from Updated2 

y_true_svm_2 = y_true_svm_2.reset_index()

temp1=pd.DataFrame()
ytemp1=pd.DataFrame()

for i in range(len(y_true_svm_2)):
    if (y_true_svm_2[0][i])!=(y_pred_svm_2[i]):
        print(i)
        ytemp1=ytemp1.append(y_updated2.loc[i], ignore_index=True)
        temp1=temp1.append(updated2.loc[i])
        updated2 = updated2.drop(i)
        y_updated2 =y_updated2.drop(i)
        
        
#========== Training for Updated2 -> M5
        
svm_model.fit(updated2, y_updated2)   
    
#========== M5 Predict UNLABELED3

svm_y_pred_f3=svm_model.predict(f3)   

#==========     Updated3 = Updated2 + Predicted UNLABELED3

updated3=pd.concat([updated2, f3], ignore_index=True, sort=False)

ypredf3=pd.DataFrame(np.array(svm_y_pred_f3))

y_updated3=pd.concat([y_updated2, ypredf3], ignore_index=True, sort=False)
#==========     Classification Updated3
print("Klasifikasi updated 3")
fiturs_train_3, fiturs_test_3, spam_nospam_u_train_3, spam_nospam_u_test_3 = train_test_split(updated3 ,y_updated3, test_size=0.3, stratify=y_updated3, random_state=20)
svm_model.fit(fiturs_train_3, spam_nospam_u_train_3)
svm_y_pred_u3=svm_model.predict(fiturs_test_3)

from sklearn import metrics
csvmsemi3 = metrics.confusion_matrix(spam_nospam_u_test_3, svm_y_pred_u3)
print(csvmsemi3)

#=pd.DataFrame(svm_y_pred)

from sklearn.metrics import classification_report
y_true_svm_3 = spam_nospam_u_test_3
y_pred_svm_3 = svm_y_pred_u3
print(classification_report(y_true_svm_3, y_pred_svm_3))

#========== Keep The True from Updated3

y_true_svm_3 = y_true_svm_3.reset_index()

temp2=pd.DataFrame()
ytemp2=pd.DataFrame()

for i in range(len(y_true_svm_3)):
    if (y_true_svm_3[0][i])!=(y_pred_svm_3[i]):
        print(i)
        ytemp2=ytemp2.append(y_updated3.loc[i], ignore_index=True)
        temp2=temp2.append(updated3.loc[i])
        updated3 = updated3.drop(i)
        y_updated3 =y_updated3.drop(i)
        

#========== Training for Updated3 -> M7

svm_model.fit(updated3, y_updated3) 

#========== M7 Predict UNLABELED4

svm_y_pred_f4=svm_model.predict(f4)

#==========     Updated4 = Updated3 + Predicted UNLABELED4

updated4=pd.concat([updated3, f4], ignore_index=True, sort=False)

ypredf4=pd.DataFrame(np.array(svm_y_pred_f4))

y_updated4=pd.concat([y_updated3, ypredf4], ignore_index=True, sort=False)
#==========     Classification Updated4
print("Klasifikasi updated 4")

fiturs_train_4, fiturs_test_4, spam_nospam_u_train_4, spam_nospam_u_test_4 = train_test_split(updated4 ,y_updated4, test_size=0.3, stratify=y_updated4, random_state=20)
svm_model.fit(fiturs_train_4, spam_nospam_u_train_4)
svm_y_pred_u4=svm_model.predict(fiturs_test_4)

from sklearn import metrics
csvmsemi4 = metrics.confusion_matrix(spam_nospam_u_test_4, svm_y_pred_u4)
print(csvmsemi4)

from sklearn.metrics import classification_report
y_true_svm_4 = spam_nospam_u_test_4
y_pred_svm_4 = svm_y_pred_u4
print(classification_report(y_true_svm_4, y_pred_svm_4))

#========== Keep The True from Updated4
y_true_svm_4 = y_true_svm_4.reset_index()

temp3=pd.DataFrame()
ytemp3=pd.DataFrame()

for i in range(len(y_true_svm_4)):
    if (y_true_svm_4[0][i])!=(y_pred_svm_4[i]):
        print(i)
        ytemp3=ytemp3.append(y_updated4.loc[i], ignore_index=True)
        temp3=temp3.append(updated4.loc[i])
        updated4 = updated4.drop(i)
        y_updated4 =y_updated4.drop(i)
        
#==========     Classification Updated4 no wrong predict left
        
fiturs_train_5, fiturs_test_5, spam_nospam_u_train_5, spam_nospam_u_test_5 = train_test_split(updated4 ,y_updated4, test_size=0.3, stratify=y_updated4, random_state=20)
svm_model.fit(fiturs_train_5, spam_nospam_u_train_5)
svm_y_pred_u5=svm_model.predict(fiturs_test_5)

from sklearn import metrics
csvmsemi5 = metrics.confusion_matrix(spam_nospam_u_test_5, svm_y_pred_u5)
print(csvmsemi5)

#=pd.DataFrame(svm_y_pred)

from sklearn.metrics import classification_report
y_true_svm_5 = spam_nospam_u_test_5
y_pred_svm_5 = svm_y_pred_u5
print(classification_report(y_true_svm_5, y_pred_svm_5))
        
print("--- %s seconds ---" % (time.time() - start_time))
'''
'''
import pickle
    
filename = 'nostem-bd1-model-poly-82-new.pickle'
pickle.dump(svm_model, open(filename, 'wb'))

tes=svm_model.coef_
'''