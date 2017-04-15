print('Script started')

import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import collections as col
import re
import random
import tensorflow as tf
import sklearn
from sklearn.preprocessing import OneHotEncoder
import csv
import time
import random

print('Modules imported')

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


# 14. Encoding cities
def enc_city(X):
    X = pd.concat([X,pd.get_dummies(X.city,prefix='city')],axis=1)
    X = X.drop('city',axis=1)
    return X

start_encode = time.time()

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
#    X = enc_city(X)    # Don't encode cities
    X = enc_slotprice(X)
    return X

def batch_balance(Y_train_s, X_train, iter_):
    sample_sz = 10000
    
    nclick_index = np.random.randint(sample_sz, size=98)
    click_index = np.where(Y_train_s[int(iter_* sample_sz/10) :int((iter_+ 1)* sample_sz/10), 0]> -0.01440388)
    click_pos_r = np.random.randint(len(click_index[0]), size=2)
    click_index_r = click_index[0][click_pos_r]
    Y_click_r = Y_train_s[click_index_r]
    X_click_r = np.asarray(X_train)[click_index_r]
    Y_nclick_r = Y_train_s[nclick_index]
    X_nclick_r = np.asarray(X_train)[nclick_index]
    
    a = np.append(Y_click_r, Y_nclick_r, axis= 0)
    b = np.append(X_click_r, X_nclick_r, axis= 0)
    p = np.random.permutation(len(a))
    
    return a[p], b[p]

def process_data(X_train):

    Y_train= pd.DataFrame(X_train[['click', 'payprice']])
    Y_train['click_payprice']= (Y_train['click'])/ Y_train['payprice']
    Y_train= Y_train.fillna(0)
    Y_train['click_payprice_s']= (Y_train['click_payprice']- Y_train['click_payprice'].mean())/ Y_train['click_payprice'].std() 
    Y_train_s= np.asarray(Y_train)[:, 3]
    Y_train_s= Y_train_s.reshape(-1, 1)
    
    X_train= encode_labels(X_train)
    X_train= enc_usertag(X_train)
    X_train= X_train.drop(['click', 'bidid', 'logtype', 'userid', 'IP', 'city', 'domain',
           'url', 'urlid', 'slotid', 'creative', 'bidprice', 'payprice',
           'keypage'], axis= 1)

    print('Data loaded. Size x/y: %s and %s' % (X_train.shape, Y_train_s.shape))
    return X_train, Y_train_s


class pCTR_MLP():
    
    def __init__(self):
        self.self = self

    def initilise_model(self):
        self.x_i = tf.placeholder("float", [None, 219])
        self.y_i = tf.placeholder("float", [None, 1])

        d_in = 219
        d_hidden1 = 500
        d_hidden2 = 500
        d_out = 1

        self.W1 = tf.Variable(tf.random_normal([d_in, d_hidden1], mean= 0.01, stddev= 0.01))
        self.b1 = tf.Variable(tf.random_normal([d_hidden1], mean= -2, stddev= 0.01))
        self.W2 = tf.Variable(tf.random_normal([d_hidden1, d_hidden2], mean= 0.01, stddev= 0.01))
        self.b2 = tf.Variable(tf.random_normal([d_hidden2], mean= -2, stddev= 0.01))
        self.W3 = tf.Variable(tf.random_normal([d_hidden2, d_out], mean= 0.01, stddev= 0.01))
        self.b3 = tf.Variable(tf.random_normal([d_out], mean= -1, stddev= 0.01))

        self.a1_i = tf.matmul(self.x_i, self.W1)+ self.b1
        self.z1_i = tf.sigmoid(self.a1_i)
        self.a2_i = tf.matmul(self.z1_i, self.W2)+ self.b2
        self.z2_i = tf.sigmoid(self.a2_i)
        self.a3_i = tf.matmul(self.z2_i, self.W3)+ self.b3
        #self.y_hat = tf.tanh(self.a3_i)
        self.y_hat = self.a3_i

        self.loss = tf.losses.mean_squared_error(self.y_hat, self.y_i)
        
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = 0.001
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                   100000, 0.96, staircase=True)
        self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) #self.learning_rate
        
    def show_var_init(self, x_array, y_array, var, n_samples):
        sess.run(tf.global_variables_initializer())
        show_dict= {self.x_i: x_array[:n_samples],
                    self.y_i: y_array[:n_samples]} 
        return sess.run(var, feed_dict= show_dict)
    
    def show_var(self, x_array, y_array, var, n_samples):
        show_dict= {self.x_i: x_array[:n_samples],
                    self.y_i: y_array[:n_samples]} 
        return sess.run(var, feed_dict= show_dict)

    def train_full(self, x_array, y_array, epoch, batch_sz):

        self.iter_= int(x_array.shape[0]/ batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        
        #try:
        #    self.load_model()
        #except:
        sess.run(tf.global_variables_initializer())

        self.train_dict= {self.x_i: x_array,
                          self.y_i: y_array} 

        for e in range(epoch):
            e_loss= 0
            if e% 100== 0:
                print('Train epoch/ epoch loss', e, e_loss)
            for i in range(self.iter_):
                i_loss= 0

                y_batch, x_batch = batch_balance(y_array, x_array, self.iter_)
                
                iter_dict= {self.x_i: x_batch,
                            self.y_i: y_batch}
                sess.run(self.optimiser, feed_dict= iter_dict)
                e_loss+= sess.run(self.loss, feed_dict= iter_dict)
            

    def save_model(self, sess):
        print('Saving model...')
        if not os.path.exists('./pCTR_NN_1model/'):
            os.mkdir('./pCTR_NN_1model/')
        saver = tf.train.Saver()
        saver.save(sess, './pCTR_NN_1model/model.checkpoint')
        print('Model saved')

    def load_model(self, sess):
        print('Loading model...')
        saver = tf.train.Saver()
        saver.restore(sess, './pCTR_NN_1model/model.checkpoint')
        print('Model loaded')

    def predict(self, x_array, y_array, save_run= False):
        #self.load_model()
        pred_dict= {self.x_i: x_array,
                    self.y_i: y_array} 

        predict_proba = sess.run(self.y_hat, feed_dict= pred_dict)
        df_predict_proba = pd.DataFrame(predict_proba)

        print(df_predict_proba[:10])

        if save_run == True:
            #output_directory = '/pCRT/Val/'
            output_filename = 'NN_pCRT_predict_proba.csv'
            df_predict_proba.to_csv('NN_pCRT_predict_proba.csv', index= False)
            print('pCTR file saved:', os.getcwd(), output_directory, output_filename)


X_train= pd.read_csv('dataset/train.csv', sep=',')
X_train= X_train[:299749]
X_valid= pd.read_csv('dataset/validation.csv', sep=',')
print('Data loaded')

X_train, Y_train_s= process_data(X_train)
X_valid, Y_valid_s= process_data(X_valid)

model = pCTR_MLP()
model.initilise_model()
with tf.Session() as sess:
    model.train_full(np.asarray(X_train), Y_train_s, 1000, 10000)
    model.save_model(sess)
    #model.load_model(sess)
    model.predict(np.asarray(X_valid), Y_valid_s, save_run= True)

print('Script ended')




