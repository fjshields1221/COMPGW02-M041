import math
import numpy as np
import os
from collections import defaultdict
from collections import Counter
import tensorflow as tf 
from sklearn import preprocessing
import pandas as pd

def load_data(X_filename, Y_filename):
	print('Importing data')
	
	df_x = pd.DataFrame()
	df_x = df_x.from_csv(X_filename)
	try:
		df_x.drop('usertag', axis=1, inplace=True)
	except:
		pass
	x_array = df_x.as_matrix()

	df_y = pd.DataFrame()
	df_y= df_y.from_csv(Y_filename)
	y_array = df_y.as_matrix()
	
	print('Data imported', x_array.shape, y_array.shape)
	return x_array, y_array


class pCTR_MLP():
    
    def __init__(self):
        self.self = self

    def initilise_model(self):
        self.x_i = tf.placeholder("float", [None, 151])
        self.y_i = tf.placeholder("float", [None, 1])

        d_in = 151
        d_hidden1 = 500
        d_hidden2 = 500
        d_out = 1

        self.W1 = tf.Variable(tf.random_normal([d_in, d_hidden1], mean= 0.01, stddev= 0.01))
        self.b1 = tf.Variable(tf.random_normal([d_hidden1], mean= -2, stddev= 0.01))
        self.W2 = tf.Variable(tf.random_normal([d_hidden1, d_hidden2], mean= 0.01, stddev= 0.01))
        self.b2 = tf.Variable(tf.random_normal([d_hidden2], mean= -2, stddev= 0.01))
        self.W3 = tf.Variable(tf.random_normal([d_hidden2, d_out], mean= 0.01, stddev= 0.01))
        self.b3 = tf.Variable(tf.random_normal([d_out], mean= -2, stddev= 0.01))

        self.a1_i = tf.matmul(self.x_i, self.W1)+ self.b1
        self.z1_i = tf.sigmoid(self.a1_i)
        self.a2_i = tf.matmul(self.z1_i, self.W2)+ self.b2
        self.z2_i = tf.sigmoid(self.a2_i)
        self.a3_i = tf.matmul(self.z2_i, self.W3)+ self.b3
        self.y_hat = tf.sigmoid(self.a3_i)

        self.loss = tf.losses.mean_squared_error(self.y_hat, self.y_i)
        self.optimiser = tf.train.GradientDescentOptimizer(0.000001).minimize(self.loss)

        self.prediction = tf.round(self.y_hat)
        self.mistakes = tf.not_equal(self.y_i, self.prediction)
        self.accuracy = 1- tf.reduce_mean(tf.cast(self.mistakes, tf.float32))

    def train_full(self, x_array, y_array, epoch, batch_sz):

        self.iter_= int(x_array.shape[0]/ batch_sz)

        try:
            self.load_model()
        except:
            sess.run(tf.global_variables_initializer())

        self.train_dict= {self.x_i: x_array,
                         self.y_i: y_array} 

        for e in range(epoch):
            e_loss= 0

            for i in range(self.iter_):
                i_loss= 0
                sta= i* batch_sz
                end= (i+ 1)* batch_sz

                iter_dict= {self.x_i: x_array[sta: end],
                            self.y_i: y_array[sta: end]}
                sess.run(self.optimiser, feed_dict= iter_dict)
                e_loss+= sess.run(self.loss, feed_dict= iter_dict)

                if i% (self.iter_/ 10)== 0:
                    print(e, i, sess.run(self.loss, feed_dict= iter_dict))

        print('Final train accuracy:', (sess.run(self.accuracy, feed_dict= self.train_dict)))

    def save_model(self, sess):
        print('Saving model...')
        if not os.path.exists('./pCTR_NNmodel/'):
            os.mkdir('./pCTR_NNmodel/')
        saver = tf.train.Saver()
        saver.save(sess, './pCTR_NNmodel/model.checkpoint')
        print('Model saved')

    def load_model(self):
        print('Loading model...')
        saver = tf.train.Saver()
        saver.restore(sess, './pCTR_NNmodel/model.checkpoint')
        print('Model loaded')

    def predict(self, x_array, y_array, save_run= True):
        self.load_model()
        pred_dict= {self.x_i: x_array,
                    self.y_i: y_array} 

        predict_proba = sess.run(self.y_hat, feed_dict= pred_dict)
        df_predict_proba = pd.DataFrame(predict_proba)

        predict = sess.run(self.prediction, feed_dict= pred_dict)
        df_predict = pd.DataFrame(predict)

        if save_run == True:
            output_directory = '/pCRT/Val/'
            output_filename = 'NN_pCRT_predict_proba.csv'
            df_predict_proba.to_csv('NN_pCRT_predict_proba.csv', index= False)
            print('pCTR file saved:', os.getcwd(), output_directory, output_filename)

x_array, y_array = load_data('X_train_features2.csv', 'Y_train_labels.csv')
x_array_val, y_array_val = load_data('X_val_features.csv', 'Y_val_labels.csv')

if __name__ == '__main__':
	model = pCTR_MLP()
	model.initilise_model()
	with tf.Session() as sess:
		model.load_model()
		model.train_full(x_array, y_array, 3, 30)
		model.predict(x_array_val, y_array_val, True)
		model.save_model(sess)
