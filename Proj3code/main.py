

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:13:35 2016

@author: arunchandrapendyala
"""


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

#BASED ON EQUATIONS IN PROJECT DESCRIPTION AND IN PGM BY CHRIS BISHOP 

######### MNIST DATA ######
import cPickle
import gzip


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
train_set , val_set, tst_set = cPickle.load(f)
f.close()


####### USPS DATA #######
from PIL import Image as im
import glob as gb
import numpy as np

count = 0


ustest = []
us_t_set = []

for i in  range(0,10):  #There are 10 folders with several images corresponding to each number
    for image in gb.glob('USPSdata/Numerals/'+ str(i) +'/*.png'):
    
        img = im.open(image)    
        new_width  = 28 # resize to 28 x 28
        new_height = 28
        
        img = img.resize((new_width, new_height))
        img_ft = np.array(img.getdata())
        ustest.append(img_ft)
        us_t_set.append(i)
        
        
    
ustest = np.array(ustest)
ustest_norm = np.divide(ustest,255,dtype=float)  #normalise the data
ustest_new = 1 - ustest_norm
us_t_set = np.array(us_t_set)

t_k_bin = np.zeros((19999,10))

for i in range(0,19999):  # true label in binary matrix form - number is represented by 1 in corresponding index
    t_k_bin[i][us_t_set[i]] = 1

import numpy as np
#<------------**************** 1. logistic regression ********************----------------->


n_len = 50000
d_len = 784  #28 x 28
k = 10 #corresponds to each digit 
t = np.zeros((n_len,1))
x = np.zeros((n_len,d_len))
#----------training------------  
x = train_set[0]
t = train_set[1]

print "############ 1. logistic regression  ###########"

t_k_bin = np.zeros((n_len,k))

for i in range(0,n_len):
    t_k_bin[i][t[i]] = 1

w_k = np.random.rand(d_len,k) # randomly assigned initial weights

b_k = 1
eta = 0.01  # learning rate


a_k_arr = np.dot(x, w_k) + b_k
y_k_bin = np.zeros((n_len,k))

a_j_sum = np.zeros((1,k))

a_exp = np.exp(a_k_arr)

a_j_sum = np.matrix.sum(np.matrix(a_exp), axis =1)

y_k_bin = a_exp/a_j_sum

w = w_k #use w for updation of weights

x_arr = np.zeros((1,784))

for j in range(0,10): 
    for i in range(0,50000):
        x_arr = np.reshape(x[i], (1,784))
       
        a_k_new =  np.dot(x_arr, w) + b_k
        a_exp_new = np.exp(a_k_new)
        a_j_sum_new = np.matrix.sum(np.matrix(a_exp_new), axis =1)
        y_k_bin = a_exp_new/a_j_sum_new
        w += np.dot(eta , np.dot( np.transpose(x_arr)  ,np.subtract(t_k_bin[i] ,y_k_bin)))

a_k_new =  np.dot(x, w) + b_k

a_exp_new = np.exp(a_k_new)  
a_j_sum_new = np.matrix.sum(np.matrix(a_exp_new), axis =1)

y_k_new = a_exp_new/a_j_sum_new

#classification and find error rate

c = np.argmax(y_k_new, axis=1)

n_corr=0
for i in range(0,50000):   # correct count
    if(c[i]==t[i]):
        n_corr += 1
print "Accuracy- training set"        
print float(n_corr)/n_len

#-------------validation----------------------------------

x = val_set[0]
t_valtn = val_set[1]

b_k_valtn = np.ones((10000,1))

a_k_valtn =  np.dot(x, w) + b_k_valtn
a_exp_valtn = np.exp(a_k_valtn) 
a_j_sum_valtn = np.matrix.sum(np.matrix(a_exp_valtn), axis =1)

y_k_valtn = a_exp_valtn/a_j_sum_valtn

#classification and find error rate

c = np.argmax(y_k_valtn, axis=1)

n_corr=0
for i in range(0,10000):
    if(c[i]==t_valtn[i]): # correct count
        n_corr += 1
print "Accuracy - validation set"        
print float(n_corr)/10000

#-------------testing----------------------------------

x = tst_set[0]
t_tst = tst_set[1]


b_k_tst = np.ones((10000,1))

a_k_valtn =  np.dot(x, w) + b_k_tst

a_exp_valtn = np.exp(a_k_valtn)  
a_j_sum_valtn = np.matrix.sum(np.matrix(a_exp_valtn), axis =1)

y_k_valtn = a_exp_valtn/a_j_sum_valtn

#classification and find error rate

c = np.argmax(y_k_valtn, axis=1)

n_corr=0
for i in range(0,10000): # correct count
    if(c[i]==t_tst[i]):
        n_corr += 1
print "Accuracy - testing set"       
print float(n_corr )/10000


#--------testing USPS data --------------

x = ustest_new
t_valtn = us_t_set

b_k_valtn = np.ones((19999,1))

a_k_valtn =  np.dot(x, w) + b_k_valtn

a_exp_valtn = np.exp(a_k_valtn) 
a_j_sum_valtn = np.matrix.sum(np.matrix(a_exp_valtn), axis =1)

y_k_valtn = a_exp_valtn/a_j_sum_valtn

#classification

c = np.argmax(y_k_valtn, axis=1)

n_corr=0
for i in range(0,19999):
    if(c[i]==t_valtn[i]):
        n_corr += 1
print "Accuracy - testing USPS data"        
print float(n_corr)/19999

#<------------**************** 2.single layer neural network ********************----------------->

n_len = 50000
m_len = 100
d_len = 784
k_len = 10
valtst_len = 10000
print "############ 2.single layer neural network ###########"
#--------------training-------------------
x = train_set[0]
t = train_set[1]

t_k_bin = np.zeros((n_len,k_len))

eta = 0.01

for i in range(0,n_len):
    t_k_bin[i][t[i]] = 1

#first part
w_ji = np.random.rand(d_len,m_len)/1000  

#second part
w_kj = np.random.rand(m_len,k_len)/1000 
b_k = 1
b_j = 1
   
for j in range(0,20):   # iterative update of weights
    for i in range(0,n_len):
         
        x_arr = np.reshape(x[i], (1,784))
        temp_j = np.dot(x_arr , w_ji) + b_j 
         
        z_j = 1/(1+np.exp(-1 * temp_j))
        
        a_k_arr = np.dot(z_j , w_kj) + b_k  
        
        a_exp = np.exp(a_k_arr)
        
        a_j_sum = np.matrix.sum(np.matrix(a_exp), axis =1)
        
        y_k_bin = a_exp/a_j_sum
        
        del_k = y_k_bin - t_k_bin[i]
        
        h_zj_der = np.dot((1-z_j),np.transpose(z_j))
        
        del_j = np.dot(h_zj_der , np.dot(del_k,np.transpose(w_kj)))
        o_e_1 = np.dot(np.transpose(x_arr),del_j)
        
        do_e_2 = np.dot(np.transpose(z_j),del_k)
        
        w_ji -= np.dot(eta,do_e_1)
        w_kj -= np.dot(eta, do_e_2)
   
temp_j = np.dot(x , w_ji) + b_j 
 
z_j = 1/(1+np.exp(-1 * temp_j))

a_k_arr = np.dot(z_j , w_kj) + b_k  

a_exp = np.exp(a_k_arr)

a_j_sum = np.matrix.sum(np.matrix(a_exp), axis =1)

y_k_bin = a_exp/a_j_sum

#classification

c = np.argmax(y_k_bin, axis=1)

n_corr=0
for i in range(0,50000):
    if(c[i]==t[i]):
        n_corr += 1
print "Accuracy - training set"       
print float(n_corr)/n_len


#-------------------validation-------------------------


x = val_set[0]
t = val_set[1]


temp_j = np.dot(x , w_ji) + b_j 
 
z_j = 1/(1+np.exp(-1 * temp_j))


a_k_arr = np.dot(z_j , w_kj) + b_k  

a_exp = np.exp(a_k_arr)

a_j_sum = np.matrix.sum(np.matrix(a_exp), axis =1)

y_k_bin = a_exp/a_j_sum

#classification

c = np.argmax(y_k_bin, axis=1)

n_corr=0
for i in range(0,valtst_len):
    if(c[i]==t[i]):
        n_corr += 1
print "Accuracy - validation set"        
print float(n_corr)/valtst_len
    
#-------------------testing-------------------------


x = tst_set[0]
t = tst_set[1]

temp_j = np.dot(x , w_ji) + b_j 
 
z_j = 1/(1+np.exp(-1 * temp_j))

a_k_arr = np.dot(z_j , w_kj) + b_k  

a_exp = np.exp(a_k_arr)


a_j_sum = np.matrix.sum(np.matrix(a_exp), axis =1)

y_k_bin = a_exp/a_j_sum

#classification

c = np.argmax(y_k_bin, axis=1)

n_corr=0
for i in range(0,valtst_len):
    if(c[i]==t[i]):
        n_corr += 1
print "Accuracy -testing set"        
print float(n_corr)/valtst_len
 
#-------------------testing USPS Data (Neural network)-------------------------


x = ustest_new
t = us_t_set


temp_j = np.dot(x , w_ji) + b_j 
 
z_j = 1/(1+np.exp(-1 * temp_j))


a_k_arr = np.dot(z_j , w_kj) + b_k  

a_exp = np.exp(a_k_arr)

a_j_sum = np.matrix.sum(np.matrix(a_exp), axis =1)

y_k_bin = a_exp/a_j_sum

#classification

c = np.argmax(y_k_bin, axis=1)

n_corr=0
for i in range(0,valtst_len):
    if(c[i]==t[i]):
        n_corr += 1
print "Accuracy - testing USPS data"       
print float(n_corr)/valtst_len
      
     
#<------------************** 3  convolutional neural network********************----------------->

#source : from slides http://jesusfbes.es/wp-content/uploads/2016/04/MLG_tensor.pdf

mnist = input_data.read_data_sets('MNIST/', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')
    
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Add dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(0,1000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    
print "test accuracy for USPS data %g"%accuracy.eval(feed_dict={
     x: ustest_new , y_: t_k_bin , keep_prob: 1.0})
