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

def import_data(train= 'No', test='Yes', val='No'):
    if train == 'Yes':
        df_train = pd.read_csv('dataset/test.csv', sep=',')
        train_lst_predict_proba= np.load('pCTR/tst_lst_predict_proba.npy')
    else:
        df_train = []
        train_lst_predict_proba = []
    if test == 'Yes':
        df_test = pd.read_csv('dataset/test.csv', sep=',')
        tst_lst_predict_proba= np.load('pCTR/tst_lst_predict_proba.npy')
    else:
        df_test = []
        tst_lst_predict_proba = []
    if val == 'Yes':
        df_val = pd.read_csv('dataset/validation.csv', sep=',')
        val_lst_predict_proba= np.load('pCTR/val_lst_predict_proba.npy')
    else:
        df_val = []
        val_lst_predict_proba = []


df_val = pd.read_csv('dataset/validation.csv', sep=',')
df_val_sub= df_val[df_val['click']== 1]
df_val_sub2= df_val[df_val['payprice']< 10]
print(len(df_val_sub2))


