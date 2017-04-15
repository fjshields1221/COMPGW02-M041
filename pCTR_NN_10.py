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
from sklearn import metrics

print('1. Loading data')
X_train= pd.read_csv('dataset/train.csv', sep=',')
X_train= X_train[:1097738]
print(len(X_train))

df_valid= pd.read_csv('dataset/validation.csv', sep=',')
X_valid= df_valid
print(len(X_valid))

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


print('2. Processing data')

X_train = encode_labels(X_train)
X_train = enc_usertag(X_train)
Y_train= pd.DataFrame(X_train[['click']])
X_train= X_train.drop(['click', 'bidid', 'logtype', 'userid', 'IP', 'city', 'domain',
       'url', 'urlid', 'slotid', 'creative', 'bidprice', 'payprice',
       'keypage'], axis= 1)
Y_train_s = np.asarray(Y_train)[:, 0]
Y_train_s= Y_train_s.reshape(-1, 1)

X_valid= encode_labels(X_valid)
X_valid= enc_usertag(X_valid)
Y_valid= pd.DataFrame(X_valid[['click']])
X_valid= X_valid.drop(['click', 'bidid', 'logtype', 'userid', 'IP', 'city', 'domain',
       'url', 'urlid', 'slotid', 'creative', 'bidprice', 'payprice',
       'keypage'], axis= 1)

Y_valid_s = np.asarray(Y_valid)[:, 0]
Y_valid_s= Y_valid_s.reshape(-1, 1)	



df_Y_train_s = pd.DataFrame(Y_train_s)
df_Y_train_s['1'] = np.where(df_Y_train_s[0]==1, 0, 1) 
Y_train_s_2 = df_Y_train_s.values

df_Y_valid_s = pd.DataFrame(Y_valid_s)
df_Y_valid_s['1'] = np.where(df_Y_valid_s[0]==1, 0, 1) 
Y_valid_s_2 = df_Y_valid_s.values


def batch_balance(X_data, Y_data_s_2, iter_):
    
    sample_sz= 10000
    index_bline = iter_* sample_sz

    nclick_index = np.random.randint(sample_sz, size=8)
    click_index = np.where(Y_train_s_2[(iter_* sample_sz): ((iter_+ 1) *sample_sz), 1]== 0)
    click_pos_r1 = np.random.randint(len(click_index[0]), size=1)
    click_pos_r2 = np.random.randint(len(click_index[0]), size=1)
    click_pos_r3 = np.random.randint(len(click_index[0]), size=1)
    click_index_r = np.append(click_index[0][click_pos_r1], click_index[0][click_pos_r2]) #, click_index[0][click_pos_r3])

    Y_click_r = Y_data_s_2[index_bline+ click_index_r]
    X_click_r = np.asarray(X_data)[index_bline+ click_index_r]
    Y_nclick_r = Y_data_s_2[index_bline+ nclick_index]
    X_nclick_r = np.asarray(X_data)[index_bline+ nclick_index]

    a = np.append(Y_click_r, Y_nclick_r, axis= 0)
    b = np.append(X_click_r, X_nclick_r, axis= 0)
    p = np.random.permutation(len(a))
    
    return b[p], a[p]


class pCTR_MLP():
    
    def __init__(self):
        self.self = self

    def initilise_model(self):
        self.x_i = tf.placeholder("float", [None, 219])
        self.y_i = tf.placeholder("float", [None, 2])

        d_in = 219
        d_hidden1 = 500
        d_hidden2 = 500
        d_out = 2

        self.W1 = tf.Variable(tf.random_normal([d_in, d_hidden1], mean= 0.01, stddev= 0.01))
        self.b1 = tf.Variable(tf.random_normal([d_hidden1], mean= -2, stddev= 0.01))
        self.W2 = tf.Variable(tf.random_normal([d_hidden1, d_hidden2], mean= 0.01, stddev= 0.01))
        self.b2 = tf.Variable(tf.random_normal([d_hidden2], mean= -2, stddev= 0.01))
        self.W3 = tf.Variable(tf.random_normal([d_hidden2, d_out], mean= 0.01, stddev= 0.01))
        self.b3 = tf.Variable(tf.random_normal([d_out], mean= -0.5, stddev= 0.01))

        self.a1_i = tf.matmul(self.x_i, self.W1)+ self.b1
        self.z1_i = tf.sigmoid(self.a1_i)
        self.a2_i = tf.matmul(self.z1_i, self.W2)+ self.b2
        self.z2_i = tf.sigmoid(self.a2_i)
        self.a3_i = tf.matmul(self.z2_i, self.W3)+ self.b3
        self.y_hat = tf.nn.softmax(self.a3_i)
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y_i, logits=self.y_hat))
        self.global_step = tf.valuesriable(0, trainable=False)
        self.starter_learning_rate = 0.0001
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                   100000, 0.96, staircase=True)
        self.optimiser = tf.contrib.keras.optimizers.Nadam(self.starter_learning_rate).minimize(self.loss) #self.learning_rate
        
        self.prediction = tf.round(self.y_hat)
        self.mistakes = tf.not_equal(self.y_i, self.prediction)
        self.accuracy = 1- tf.reduce_mean(tf.cast(self.mistakes, tf.float32))
        
    def show_var_init(self, x_array, y_array, iter_= 10):
        sess.run(tf.global_variables_initializer())
        self.show_dict= {self.x_i: x_array[:1000],
                         self.y_i: y_array[:1000]} 
        print('Pre op:', sess.run(self.y_hat, feed_dict= self.show_dict))
        for i in range(iter_):
            sess.run(self.optimiser, feed_dict= self.show_dict)
        print('Post op:', sess.run(self.y_hat, feed_dict= self.show_dict))
        
    def show_var(self, x_val, y_val):
        print('y_hat:', sess.run(self.y_hat, feed_dict= self.iter_dict))
        print('y_i:', sess.run(self.y_i, feed_dict= self.iter_dict))
        print('loss:', sess.run(self.loss, feed_dict= self.iter_dict))
        print('accuracy:', sess.run(self.accuracy, feed_dict= self.iter_dict))
        
        x_array_b, y_array_b = batch_balance(x_val, y_val, 0)
        self.valid_dict= {self.x_i: x_array_b,
                          self.y_i: y_array_b} 
        print('accuracy:', sess.run(self.accuracy, feed_dict= self.valid_dict))
        

    def train_full(self, x_array, y_array, epoch, batch_sz):

        self.iter_= int(x_array.shape[0]/ batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        
        sess.run(tf.global_variables_initializer())
        
        self.train_dict= {self.x_i: x_array,
                          self.y_i: y_array} 
        
        if self.iter_ % (self.iter_/10)== 0:
            print(sess.run(self.loss, feed_dict= train_dict))
        
        for e in range(epoch):
            e_loss= 0
                
            for i in range(self.iter_):
                iter_dict= {self.x_i: x_array[(i* batch_sz):((i+ 1)* batch_sz)],
                            self.y_i: y_array[(i* batch_sz):((i+ 1)* batch_sz)]}
                
                sess.run(self.optimiser, feed_dict= iter_dict)
                e_loss+= sess.run(self.loss, feed_dict= iter_dict)
                
                if e% 1000== 0:
                    print('Epoch %s, loss %s' % (e, sess.run(self.loss, feed_dict= iter_dict)))
            
    def train_full_bal(self, x_array, y_array, epoch, batch_sz):

        self.iter_= int(x_array.shape[0]/ 10000)- 1
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        
        sess.run(tf.global_variables_initializer())
        
        self.train_dict= {self.x_i: x_array,
                          self.y_i: y_array} 
        
        if self.iter_ % (self.iter_/10)== 0:
            print(sess.run(self.loss, feed_dict= self.train_dict))
        
        for e in range(epoch):
            e_loss= 0
                
            for i in range(self.iter_):
                
                x_array_b, y_array_b = batch_balance(x_array, y_array, i)
                self.x_array_b = x_array_b
                self.y_array_b = y_array_b
                
                self.iter_dict= {self.x_i: x_array_b,
                                 self.y_i: y_array_b}
                
                sess.run(self.optimiser, feed_dict= self.iter_dict)
                e_loss+= sess.run(self.loss, feed_dict= self.iter_dict)
                #print('Epoch %s, iter %s, loss %s' % (e, self.iter_ ,sess.run(self.loss, feed_dict= iter_dict)))
                #print(sess.run(self.y_i, feed_dict= self.iter_dict))
                
            if e% 10== 0:
                print('Epoch %s, iter %s, loss %s' % (e, self.iter_ ,sess.run(self.loss, feed_dict= self.iter_dict)))
            
    def predict(self, x_array, y_array, save_run= False):
        pred_dict= {self.x_i: x_array,
                    self.y_i: y_array} 

        predict_proba = sess.run(self.y_hat, feed_dict= pred_dict)
        self.df_predict_proba = pd.DataFrame(predict_proba)

        if save_run == True:
            output_directory = '/pCRT/'
            output_filename = 'NN2_pCTR_validation.csv'
            df_predict_proba.to_csv('NN2_pCTR_validation.csv', index= False)
            print('pCTR file saved:', os.getcwd(), output_directory, output_filename)
            
    def test_accuracy(self, df_valid):
        fpr, tpr, thresholds = metrics.roc_curve([click for click in df_valid.click.values], self.df_predict_proba[0].values)
        print('AUC accuracy:', metrics.auc(fpr, tpr))
        print('Pointwise accuracy', sess.run(self.accuracy, feed_dict= self.train_dict))
        if metrics.auc(fpr, tpr)> 0.829294389288:
            print('LR accuracy beat!')
            
    def save_model(self, sess):
        print('Saving model...')
        if not os.path.exists('./pCTR_NNmodel/'):
            os.mkdir('./pCTR_NNmodel/')
        saver = tf.train.Saver()
        saver.save(sess, './pCTR_NNmodel/model.checkpoint')
        print('Model saved')

    def load_model(self, sess):
        print('Loading model...')
        saver = tf.train.Saver()
        saver.restore(sess, './pCTR_NNmodel/model.checkpoint')
        print('Model loaded')

print('3. Running NN')

model = pCTR_MLP()
model.initilise_model()
with tf.Session() as sess:
    model.train_full_bal(np.asarray(X_train), Y_train_s_2, 1, 10)
    model.predict(np.asarray(X_valid), Y_valid_s_2)
    model.test_accuracy(df_valid)    
    model.predict(np.asarray(X_train), Y_train_s_2)
    model.test_accuracy(df_train)

