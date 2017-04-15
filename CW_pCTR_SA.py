import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import collections as col
import re
import random
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
#from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import auc,roc_curve

def load_data(train='Yes', test='No', validation='No'):
	"""Loads and returns datasets as required
	   Return empty lst for if 'No'
	"""
	if train=='Yes':
		df_train = pd.read_csv('dataset/train.csv', sep=',')
	else:
		df_train = []

	if test=='Yes':
		df_test = pd.read_csv('dataset/test.csv', sep=',')
	else:
		df_test = []

	if validation=='Yes':
		df_validation = pd.read_csv('dataset/validation.csv', sep=',')
	else:
		df_validation = []
	
	print('Data loaded', len(df_train), len(df_test), len(df_validation))
	return pd.DataFrame(df_train), pd.DataFrame(df_test), pd.DataFrame(df_validation)


# Handling categorical data with one hot encoding

# 1. Encode day of week
def enc_day(X):
    X = pd.concat([X,pd.get_dummies(X.weekday,prefix='day')],axis=1)
    X = X.drop('weekday',axis=1)
    return X

# 2. Encode hours
def enc_hrs(X):
    X = pd.concat([X,pd.get_dummies(X.hour,prefix='hour')],axis=1)
    X = X.drop('hour',axis=1)
    return X

# Split user agent into 2 ~ OS and browser
def enc_OS_browser(X):
    df = pd.DataFrame(X.useragent.str.split('_',1).tolist(),
                                   columns = ['OS','browser'])
    X = pd.concat([X,df],axis=1)

    # 3. Encode OS
    X = pd.concat([X,pd.get_dummies(X.OS,prefix='OS')],axis=1)
    X = X.drop('OS',axis=1)

    # 4. Encode browser
    X = pd.concat([X,pd.get_dummies(X.browser,prefix='browser')],axis=1)
    X = X.drop('browser',axis=1)
    
    X = X.drop('useragent',axis=1)
    return X

# 5. Encode region
def enc_region(X):
    X = pd.concat([X,pd.get_dummies(X.region,prefix='region')],axis=1)
    X = X.drop('region',axis=1)
    return X

# 6. Encode adexchange
def enc_adexchange(X):
    X = pd.concat([X,pd.get_dummies(X.adexchange,prefix='adexchange')],axis=1)
    X = X.drop('adexchange',axis=1)
    return X

# 7. Encode slotwidth
def enc_slotwidth(X):
    X = pd.concat([X,pd.get_dummies(X.slotwidth,prefix='slotwidth')],axis=1)
    X = X.drop('slotwidth',axis=1)
    return X

# 8. Encode slotheight
def enc_slotheight(X):
    X = pd.concat([X,pd.get_dummies(X.slotheight,prefix='slotheight')],axis=1)
    X = X.drop('slotheight',axis=1)
    return X

# 9. Encode slotvisibility
def enc_slotvisibility(X):
    X = pd.concat([X,pd.get_dummies(X.slotvisibility,prefix='slotvisibility')],axis=1)
    X = X.drop('slotvisibility',axis=1)
    return X

# 10. Encode slotformat
def enc_slotformat(X):
    X = pd.concat([X,pd.get_dummies(X.slotformat,prefix='slotformat')],axis=1)
    X = X.drop('slotformat',axis=1)
    return X

# 11. Encode advertiser
def enc_advertiser(X):
    X = pd.concat([X,pd.get_dummies(X.advertiser,prefix='advertiser')],axis=1)
    X = X.drop('advertiser',axis=1)
    return X

# 12. Encoding slotprice into buckets
def enc_slotprice(X):
    bins = pd.DataFrame()
    bins['slotprice_bins'] = pd.cut(X.slotprice.values,5, labels=[1,2,3,4,5])

    X = pd.concat([X,bins],axis=1)
    X = pd.concat([X,pd.get_dummies(X.slotprice_bins,prefix='slotprice')],axis=1)

    X = X.drop('slotprice',axis=1)
    X = X.drop('slotprice_bins',axis=1)
    bins.pop('slotprice_bins')
    return X

# 13. Encoding user tags
def enc_usertag(X):
    a = pd.DataFrame(X.usertag.str.split(',').tolist())
    usertag_df = pd.DataFrame(a)
    usertag_df2 = pd.get_dummies(usertag_df,prefix='usertag')
    usertag_df2 = usertag_df2.groupby(usertag_df2.columns, axis=1).sum()
    X = pd.concat([X, usertag_df2], axis=1)
    X = X.drop('usertag', axis=1)
    return X

def encode_labels(X):
    X = enc_day(X)
    X = enc_hrs(X)
    X = enc_OS_browser(X)
    X = enc_region(X)
    X = enc_adexchange(X)
    X = enc_slotwidth(X)
    X = enc_slotheight(X)
    X = enc_slotvisibility(X)
    X = enc_slotformat(X)
    X = enc_advertiser(X)
    X = enc_slotprice(X)
    return X



df_train, df_test, df_validation= load_data('Yes', 'No', 'Yes')

def split_data(df_data):
	X = df_data.drop(['click','bidid','logtype','userid','IP','domain','url','urlid','slotid','creative','bidprice','payprice','keypage'], axis=1)
	Y = df_data.click
	return X, Y

def split_data_test(df_data):
	X = df_data.drop(['bidid','logtype','userid','IP','domain', 'url','urlid','slotid','creative','keypage'], axis=1)
	return X

X_train, Y_train = split_data(df_train)
X_val, Y_val = split_data(df_validation)
#X_test, y_test = split_data_test(df_test)

print('1')

X_train = encode_labels(X_train)
X_train = enc_usertag(X_train)

print('2')

X_val = enc_usertag(X_val)
X_val = encode_labels(X_val)

print('3')

clf_l2_LR = LogisticRegression(class_weight='balanced')
y_pred = clf_l2_LR.fit(X_train, Y_train).predict(X_val)
predprobs = clf_l2_LR.predict_proba(X_val)
pCTR = pd.DataFrame(predprobs)

print('4')

pred =[]
a = len(train) / 2 * np.bincount(train.click)
w = a[1] / a[0]

for p in pCTR[1]:
    pred.append( p / (p + ((1-p)/w)))

print('5')

fpr, tpr, thresholds = metrics.roc_curve([click for click in validation.click], pred)
print('AUC accuracy:', metrics.auc(fpr, tpr))

output_directory= 'Submission/Val/'
output_filename= 'Group_XX.csv'
pred.to_csv(output_directory + output_filename)
print('Submission file saved: ', os.getcwd(), output_directory, output_filename)







