print('---Script start---')

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

# Learn
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
	return df_train, df_test, df_validation

dict_label_encoder = {}
def label_encoder(df_column_nint, column_le= None): 
	"""df_column_nint in form df_data[column_name]
	"""
	if column_le== None:
		column_le = LabelEncoder()
		column_le.fit(df_column_nint.unique())
	if column_le!= None:
		pass
	df_column_int = column_le.transform(df_column_nint)
	
	dict_label_encoder[df_column_nint.columns] = column_le
	
	return df_column_int

def label_decoder(df_column_int, column_le):
	pass


def build_NB_model(df_train, lst_int_features, lst_nint_features):
	"""Format, label encode data and build NB model for specific columns 
	   Return NB_model
	"""
	# y
	array_y = df_train[['click']].as_matrix()
	array_y_r = np.reshape(array_y, (-1, 1))

	# x (int features)
	array_x_i = df_train[lst_int_features].as_matrix()

	# Model
	NB_model = GaussianNB()
	NB_model.fit(array_x_i, array_y_r)

	return NB_model 

def pred_NB_model(NB_model, df_test):
	array_bid = df_test[['bidid']].as_matrix()
	array_x_i = df_test[lst_int_features].as_matrix()

	lst_predict_log_proba = []
	lst_predict = []
	lst_predict_proba = []

	lst_predict_log_proba = np.column_stack((array_bid, NB_model.predict_log_proba(array_x_i)))
	lst_predict = np.column_stack((array_bid, NB_model.predict(array_x_i)))
	lst_predict_proba =  np.column_stack((array_bid, NB_model.predict_proba(array_x_i)))

	return lst_predict_log_proba, lst_predict, lst_predict_proba

def test_NB_model():
	pass


df_train, df_test, df_validation= load_data('Yes', 'Yes', 'Yes')

lst_int_features = ['weekday', 'hour', 'slotwidth', 'slotheight', 'advertiser'] # 'region', 'city'
lst_nint_features = ['adexchange', 'slotformat', 'slotvisibility', 'useragent']

NB_model= build_NB_model(df_train, lst_int_features, lst_nint_features)

val_lst_predict_log_proba, val_lst_predict, val_lst_predict_proba= pred_NB_model(NB_model, df_validation)
np.save('pCTR/val_lst_predict_log_proba', val_lst_predict_log_proba)
np.save('pCTR/val_lst_predict', val_lst_predict)
np.save('pCTR/val_lst_predict_proba', val_lst_predict_proba)

tst_lst_predict_log_proba, tst_lst_predict, tst_lst_predict_proba= pred_NB_model(NB_model, df_test)
np.save('pCTR/tst_lst_predict_log_proba', tst_lst_predict_log_proba)
np.save('pCTR/tst_lst_predict', tst_lst_predict)
np.save('pCTR/tst_lst_predict_proba', tst_lst_predict_proba)

print('---Script end---')




