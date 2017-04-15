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

# Import models and datasets (containing pCTR)
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

    print('Data imported')
    return df_train, train_lst_predict_proba, df_test, tst_lst_predict_proba, df_val, val_lst_predict_proba

# Baseline model
def static_bid_val(df_data, bid, random_bid= False, print_YN= False, test_run= False):
    """
    """
    total_cost = 0
    budget = 250000
    max_bids = 299749
    
    if test_run== False:
        
        total_cost = 0
        budget = 250000
        imp_w = 0
        imp_l = 0
        clicks = 0
        lst_win = {}

        if random_bid== False:
            our_bid= bid
        else:
            max_bid= bid
        
        df_data_sub = df_data[['bidid', 'click', 'payprice']]

        for bidid, click, payprice in df_data_sub.values:
            
            if (total_cost+ payprice) > budget:
                #print('Max budget reached')
                break
            if (imp_w+ imp_l) == max_bids:
                print('Max imps reached')
                #break

            if random_bid== True:
                imp_bid= random.randint(0, bid)
            else:
                imp_bid= bid
                
            if payprice < imp_bid: #and click > 0
                imp_w += 1
                clicks += click
                total_cost += payprice
                lst_win[bidid]= [payprice, click]
                
            if payprice >= imp_bid:
                imp_l += 1
                
        imp_total = len(df_data)
        try:
            imp_comp = round((imp_w+ imp_l)/ imp_total, 3)
        except: 
            imp_comp = 0
        try:
            bud_spent = total_cost/ budget
        except: 
            bud_spent = 0        
        try:
            CTR = round((clicks/ imp_w), 5)
        except:
            CTR = 0
        
        if print_YN== True:
            #print('Baseline bid:', imp_bid)
            #print('Bids won:', imp_w)
            #print('Bids lost:', imp_l)
            #print('% bids comp', imp_comp)
            #print('Clicks:', clicks)
            #print('CTR:', CTR)
            #print('Total cost:', total_cost)
            print('Baseline bid:', imp_bid, 'Clicks:', clicks, '% bids comp', imp_comp, 'Total cost:', total_cost)
        
        # Evaluation 
        return clicks, CTR, imp_w, imp_comp, bud_spent

    if test_run==True:
        lst_id_bid = []

        if random_bid== True:
            imp_bid= random.randint(0, bid)
        else:
            imp_bid= bid

        df_data_sub = df_data[['bidid']]

        for bidid in df_data_sub.values:
            lst_id_bid.append((str(bidid),imp_bid))
            if len(lst_id_bid) >= max_bids:
                print('Max imps reached')
                break

        print('Test done')

        # Submission 
        return lst_id_bid


# Improved model with pCTR
def dynamic_bid_val(df_data, pCTR_data, baseline_bid, print_YN='Yes', test_run= False):
    
    #try:
    #    assert(len(df_data)==len(pCTR_data))
    #    print('Data load successful')
    #except:
    #    print('Data load fail')

    ave_pCTP = np.mean(pCTR_data[:,2])
    val_pCTR_norm = pCTR_data[:,2]/ ave_pCTP

    if test_run== True:
        
        lst_id_bid = []

        df_data_sub = np.column_stack((df_data[['bidid']].values, val_pCTR_norm))

        for i in df_data_sub:
            imp_bid = baseline_bid* i[1]
            lst_id_bid.append((str(i[0]), imp_bid))
            
            #if len(lst_id_bid) >= max_bids:
            #    print('Max imps reached')
            #    break

        return lst_id_bid
        print('Test done')
    
    if test_run== False:

        total_cost = 0
        budget = 250000
        imp_w = 0
        imp_l = 0
        clicks = 0
        lst_win= []

        df_data_sub = np.column_stack((df_data[['bidid', 'click', 'payprice']].values, val_pCTR_norm))
        
        for i in df_data_sub:
            if (total_cost + i[2]) > budget: # 'payprice'
                break
            
            our_bid = baseline_bid* i[3] # pCTR
            
            if i[2] < our_bid: 
                imp_w += 1
                clicks += i[1] # 'clicks'
                total_cost += i[2] # 'payprice'
                
            if i[2] >= our_bid:
                imp_l += 1
        
        imp_total = len(df_data)
        try:
            imp_comp = round((imp_w+ imp_l)/ imp_total, 3)
        except: 
            imp_comp = 0
        try:
            bud_spent = total_cost/ budget
        except: 
            bud_spent = 0        
        try:
            CTR = round((clicks/ imp_w), 5)
        except:
            CTR = 0
        
        if print_YN== 'Yes':
            #print('Baseline bid:', baseline_bid)
            #print('Bids won:', imp_w)
            #print('Bids lost:', imp_l)
            #print('% bids comp', imp_comp)
            #print('Clicks:', clicks)
            #print('CTR:', CTR)
            #print('Total cost:', total_cost)
            print('Baseline bid:', baseline_bid, 'Clicks:', clicks, '% bids comp', imp_comp, 'Total cost:', total_cost)
        
        return clicks, CTR, imp_w, imp_comp, bud_spent



def cross_validation(df_data, model_type, random_bid, pCTR_data=[]):
    """ CV across training data for data len 299749== df_test. 
        Currently 1/8 CV
        Return dict with average clicks for each CV iter
    """
    model_length = len(df_data)
    iter_ = int(299749/ model_length) #int(len(df_train)/ model_length)
    
    iter_CTR_dict = {}
    # Iterate through bidprice range
    for i in range(10,30,1):    
        max_clicks = 0
        max_clicks_iter = 0
        
        iter_click_lst = []
        for j in range(iter_):
            sta= j* model_length
            end= (j+ 1)* model_length
            df_iter = df_data.iloc[sta: end]

            if model_type== 'static_bid_val':
                clicks, CTR, imp_w, imp_comp, bud_spent= static_bid_val(df_iter, i, random_bid, True, test_run= False)

            if model_type== 'dynamic_bid_val':
                clicks, CTR, imp_w, imp_comp, bud_spent = dynamic_bid_val(df_iter, pCTR_data, i, print_YN='Yes', test_run=False)

            iter_click_lst.append(clicks)
            
        iter_CTR_dict[i]= np.average(iter_click_lst)
    
    # Find optimal bidprice
    for i, j in iter_CTR_dict.items():
        if j == max(iter_CTR_dict.values()):
            opt_our_bid = i
    
    return iter_CTR_dict, opt_our_bid

# Run model
df_train, train_lst_predict_proba, df_test, tst_lst_predict_proba, df_val, val_lst_predict_probas = import_data(train= 'No', test='No', val='Yes')

# Q2
# a. Constant bid
print('Constant bid')
#clicks, CTR, imp_w, imp_comp, bud_spent = static_bid_val(df_val, 10, False, True, test_run= False)
iter_CTR_dict, opt_our_bid= cross_validation(df_val, 'static_bid_val', random_bid=False)
#lst_id_bid = static_bid_val(df_val, 10, False, True, test_run= True)

# b. Random bid
#clicks, CTR, imp_w, imp_comp, bud_spent = static_bid_val(df_val, 10, True, True, test_run= False)
#iter_CTR_dict, opt_our_bid= cross_validation(df_val, 'static_bid_val', random_bid=True)
#lst_id_bid = static_bid_val(df_val, 10, True, True, test_run= True)

# Q3
print('Linear w/ pCTR bid')
# a. Optimise dynamic_bid_val()
# a.1 for baseline bid
pCTR_val = np.load('pCTR/val_lst_predict_proba.npy')
iter_CTR_dict, opt_our_bid= cross_validation(df_val, 'dynamic_bid_val', False, pCTR_val)

# a.2 for features in NB model


# b. Submission file
#pCTR_val = np.load('pCTR/val_lst_predict_proba.npy')
#clicks, CTR, imp_w, imp_comp, bud_spent = dynamic_bid_val(df_val, pCTR_val, 100, print_YN='Yes', test_run= True)
#lst_id_bid = dynamic_bid_val(df_val, pCTR_val, 100, print_YN='Yes', test_run= False)


print('End script')



