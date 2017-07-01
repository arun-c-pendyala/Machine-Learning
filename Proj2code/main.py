# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 23:02:11 2016

@author: arunchandrapendyala
"""

import numpy as np

print("#######LETOR DATA#######")

file = open("Querylevelnorm.txt");

rel=[]
x_arr = []
for line in file:
    ft =[]
    first = True
    lines = line.split()
    for w in lines:
        if first:                    #the first col contains the relevance value
            rel.append(int(w))
            first = False
        elif w.startswith('qid'):
            pass
        elif w.startswith('#'):     #ignore comments and qid values
            break
        else:
            n = float(w.split(':')[1])            
            ft.append(n)
    x_arr.append(ft)
        

x_mat = np.matrix(x_arr)
x_nparray = np.array(x_arr)
rel_arr = np.array(rel)

letor_size = 69623
letor_train_size = 55699
letor_val_size = 6962
letor_tst_size = 6962
letor_ft_len = 46


#calculation of sigma inverse


x_train = np.zeros((letor_train_size,letor_ft_len))

for i in range(0,letor_train_size):
    x_train[i] = x_nparray[i]


var_mat_train = np.zeros((letor_ft_len,letor_ft_len))

for i in range(0,letor_ft_len):
    var_temp = np.var(x_train[[0,letor_train_size-1],i])

    var =  (var_temp)/1000
    if(var != 0.000):  
        var_mat_train[i][i] = var
    else:
        var_mat_train[i][i] = 0.00001
        
    

var_invmat_train = np.linalg.inv(var_mat_train)
m = 7

mu_matrix = np.zeros((m,letor_ft_len))



for i in range(1,m):
    rand = np.random.randint(0,letor_train_size)
    #print(rand)
    #randno = rand[0]
    for j in range(0,letor_ft_len):
        mu_matrix[i][j] = x_nparray[rand][j]
        
rel_train = np.zeros(letor_train_size)
        
for i in range(0,letor_train_size):
    rel_train[i] = rel_arr[i]
    
    
'''
def phi_fn(i, mu_mat, var_invmat, x_arr):
    
    
    phifn = np.exp(-0.5 * np.dot((np.subtract(x_arr[i,:],mu_mat),np.dot(var_invmat,((np.subtract(x_arr[i,:],mu_mat)).transpose)))))   
    return phifn
'''
phi_matrix_train = np.zeros((letor_train_size,m))

mu_temp = []
x_temp = []
diff = []
lmda_ml  = 0.001
lmda_mat = lmda_ml *np.identity(m) 



for i in range(0,letor_train_size):
    
    for j in range(0,m):
        
        if (j==0):
            phi_matrix_train[i][j] = 1
        else:
            
            diff = np.subtract(x_nparray[i] , mu_matrix[j])
            
            phi_matrix_train[i][j] =  (np.exp(np.dot(-0.5 , np.dot(  np.dot(diff,var_invmat_train),(np.transpose(diff))))))
            
#print(phi_matrix)       
w_ml_st = np.dot(np.dot(np.linalg.inv(np.add(lmda_mat ,np.dot((np.transpose(phi_matrix_train)) , phi_matrix_train))),(np.transpose(phi_matrix_train))),rel_train)
print("weights(CFS)")
print(w_ml_st)


    
sum_sq = 0

for i in range(0,letor_train_size):
    sq_diff = np.square(rel_train[i] - np.dot(w_ml_st , phi_matrix_train[i]))
    
    sum_sq += sq_diff
    
    
err_dw = np.dot(sum_sq , 0.5)


err_w = err_dw + lmda_ml * 0.5 * np.dot(np.transpose(w_ml_st), w_ml_st)


err_rms_train = np.sqrt(2*err_w/letor_train_size)
print("RMS error-training error(CFS)")
print(err_rms_train)

#validation set


rel_valtn = np.zeros(letor_val_size)

for i in range(letor_train_size,62661):
    rel_valtn[i-letor_train_size] = rel_arr[i]
  
x_valtn = np.zeros((letor_val_size,letor_ft_len))
  
for i in range(letor_train_size,62661):
    x_valtn[i-letor_train_size] = x_nparray[i]
  
x_tst = np.zeros((letor_tst_size,letor_ft_len))  
for i in range(62661,letor_size):
    x_tst[i-letor_size] = x_nparray[i]
    
phi_matrix_valtn = np.zeros((letor_val_size,m))
 
for i in range(0,letor_val_size):
    
    for j in range(0,m):
        
        if (j==0):
            phi_matrix_valtn[i][j] = 1
        else:
            
            diff = np.subtract(x_valtn[i] , mu_matrix[j])
            
            phi_matrix_valtn[i][j] =  (np.exp(np.dot(-0.5 , np.dot(  np.dot(diff,var_invmat_train),(np.transpose(diff))))))
            

phi_matrix_tst = np.zeros((letor_tst_size,m))
 
for i in range(0,letor_val_size):
    
    for j in range(0,m):
        
        if (j==0):
            phi_matrix_tst[i][j] = 1
        else:
            
            diff = np.subtract(x_tst[i] , mu_matrix[j])
            
            phi_matrix_tst[i][j] =  (np.exp(np.dot(-0.5 , np.dot(  np.dot(diff,var_invmat_train),(np.transpose(diff))))))
            
    
err_w = []
sum_sq = 0

for i in range(0,letor_val_size):
    sq_diff = np.square(rel_valtn[i] - np.dot(w_ml_st , phi_matrix_valtn[i]))
    
    sum_sq += sq_diff
    
    
err_dw_valtn = np.dot(sum_sq , 0.5)



err_rms_valtn = np.sqrt(2*err_dw_valtn/letor_val_size)
print("RMS error-validation error(CFS)")
print(err_rms_valtn)

#testing

rel_valtn = np.zeros(6962)

for i in range(62661,letor_size):
    rel_valtn[i-62661] = rel_arr[i]
  

err_w = []
sum_sq = 0

for i in range(0,6261):
    sq_diff = np.square(rel_valtn[i] - np.dot(w_ml_st , phi_matrix_tst[i]))
    
    sum_sq += sq_diff
    
    
err_dw_valtn = np.dot(sum_sq , 0.5)



err_rms_valtn_syn = np.sqrt(2*err_dw_valtn/6261)
print("RMS error-testing error(CFS)")
print(err_rms_valtn_syn)


#stochastic gradient descent

w_sgd_new = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]

w_sgd_iter = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]


lmda = lmda_ml
eta = 0.001
for i in range(0, letor_train_size):
    delta_ED = np.dot(np.subtract( rel_train[i] , np.dot(np.transpose(w_sgd_iter) , phi_matrix_train[i])) , phi_matrix_train[i])

    del_w = np.dot(eta, np.subtract(delta_ED , np.dot(lmda, w_sgd_iter)))
    w_sgd_new = w_sgd_iter + del_w 
    w_sgd_iter = w_sgd_new
print("weights(SGD)")
print(w_sgd_iter)
err_w_sgd = []
sum_sq_sgd = 0

for i in range(0,letor_train_size):
    sq_diff_sgd = np.square(rel_train[i] - np.dot(w_sgd_new , phi_matrix_train[i]))
    
    sum_sq_sgd += sq_diff_sgd
    
    
err_dw_sgd = np.dot(sum_sq_sgd , 0.5)

err_rms_train_sgd = np.sqrt(2*err_dw_sgd/letor_train_size)
print("RMS error - training error (SGD)")
print(err_rms_train_sgd)

#validation set


rel_valtn_sgd = np.zeros(letor_val_size)

for i in range(letor_train_size,62661):
    rel_valtn_sgd[i-letor_train_size] = rel_arr[i]
  
x_valtn_sgd = np.zeros((letor_val_size,letor_ft_len))
  
for i in range(letor_train_size,62661):
    x_valtn_sgd[i-letor_train_size] = x_nparray[i]
    

    
err_w = []
sum_sq_sgd = 0

for i in range(0,letor_val_size):
    sq_diff_sgd = np.square(rel_valtn[i] - np.dot(w_sgd_new , phi_matrix_valtn[i]))
    
    sum_sq_sgd += sq_diff_sgd
    
    
err_dw_valtn_sgd = np.dot(sum_sq_sgd , 0.5)



err_rms_valtn_sgd = np.sqrt(2*err_dw_valtn_sgd/letor_val_size)
print("RMS error-validation error(SGD)")
print(err_rms_valtn_sgd)

#testing

rel_valtn = np.zeros(6962)

for i in range(62661,letor_size):
    rel_valtn[i-62661] = rel_arr[i]
  

err_w = []
sum_sq = 0

for i in range(0,6261):
    sq_diff = np.square(rel_valtn[i] - np.dot(w_sgd_new , phi_matrix_tst[i]))
    
    sum_sq += sq_diff
    
    
err_dw_valtn = np.dot(sum_sq , 0.5)



err_rms_valtn_syn = np.sqrt(2*err_dw_valtn/6261)
print("RMS error-testing error(SGD)")
print(err_rms_valtn_syn)

print("Hyperparameters")

print("M")
print(m)
print("lambda")
print(lmda_ml)
print("eta")
print(eta)

####################################

import csv
import numpy as np


print("#######SYNTHETIC DATA#######")

x_arr_syn = []
out_arr_syn = []

with open('input.csv', 'rU') as infile:
    inreader = csv.reader(infile)
    for row in inreader:
        ft_syn = []
    
        for i in range(0,10):
            ft_syn.append(float(row[i]))
        x_arr_syn.append(ft_syn)
        
x_mat_syn = np.matrix(x_arr_syn)
in_syn = np.array(x_arr_syn)   #input data array   

with open('output.csv', 'rU') as outfile:
    outreader = csv.reader(outfile)
    for row in outreader:
        
        out_arr_syn.append(int(row[0]))
        
out_mat_syn = np.matrix(out_arr_syn)
out_syn = np.array(out_arr_syn)  #output data array(t)

syn_size = 20000
syn_train_size = 16000
syn_val_size = 2000
syn_tst_size = 2000
ft_len = 10

#calculation of sigma inverse


x_train_syn = np.zeros((syn_train_size,ft_len))

for i in range(0,syn_train_size):
    x_train_syn[i] = in_syn[i]


var_mat_train_syn = np.zeros((ft_len,ft_len))

for i in range(0,ft_len):
    var_temp = np.var(x_train_syn[[0,syn_train_size-1],i])

    var =  (var_temp)/10
    if(var != 0.000):  
        var_mat_train_syn[i][i] = var
    else:
        var_mat_train_syn[i][i] = 0.00001
        
    

var_invmat_train_syn = np.linalg.inv(var_mat_train_syn)
m = 7

mu_matrix = np.zeros((m,ft_len))

# 0.78 for m =100 , 0.7658032 for m = 1000 , 0.581026908507 for m = 10000

for i in range(1,m):
    rand = np.random.randint(0,syn_train_size)
    #print(rand)
    #randno = rand[0]
    for j in range(0,ft_len):
        mu_matrix[i][j] = in_syn[rand][j]
        
rel_train_syn = np.zeros(syn_train_size)
        
for i in range(0,syn_train_size):
    rel_train_syn[i] = out_syn[i]
    
    

phi_matrix_train_syn = np.zeros((syn_train_size,m))

mu_temp = []
x_temp = []
diff = []
lmda_ml_syn  = 0.001

lmda_mat_syn = lmda_ml_syn *np.identity(m) 



for i in range(0,syn_train_size):
    
    for j in range(0,m):
        
        if (j==0):
            phi_matrix_train_syn[i][j] = 1
        else:
            
            diff = np.subtract(in_syn[i] , mu_matrix[j])
            
            phi_matrix_train_syn[i][j] =  (np.exp(np.dot(-0.5 , np.dot(  np.dot(diff,var_invmat_train_syn),(np.transpose(diff))))))
            
#print(phi_matrix)       
w_ml_st_syn = np.dot(np.dot(np.linalg.inv(np.add(lmda_mat_syn ,np.dot((np.transpose(phi_matrix_train_syn)) , phi_matrix_train_syn))),(np.transpose(phi_matrix_train_syn))), rel_train_syn)
print("weights(cfs)")
print(w_ml_st_syn)


    
sum_sq = 0

for i in range(0,syn_train_size):
    sq_diff = np.square(rel_train_syn[i] - np.dot(w_ml_st_syn , phi_matrix_train_syn[i]))
    
    sum_sq += sq_diff
    
    
err_dw_syn = np.dot(sum_sq , 0.5)


err_w_syn = err_dw_syn + lmda_ml_syn * 0.5 * np.dot(np.transpose(w_ml_st_syn), w_ml_st_syn)


err_rms_train_syn = np.sqrt(2*err_w_syn/syn_train_size)
print("RMS error- training error(CFS)")
print(err_rms_train_syn)


#validation set

rel_valtn_syn = np.zeros(syn_val_size)

for i in range(syn_train_size,18000):
    rel_valtn_syn[i-syn_train_size] = out_syn[i]
  
x_valtn_syn = np.zeros((syn_train_size,ft_len))
  
for i in range(syn_train_size,18000):
    x_valtn_syn[i- syn_train_size] = in_syn[i]
    
x_tst_syn = np.zeros((syn_train_size,ft_len))
  
for i in range(18000,20000):
    x_tst_syn[i- syn_train_size] = in_syn[i]
    
phi_matrix_valtn_syn = np.zeros((syn_val_size,m))
    
for i in range(0,syn_val_size):
    
    for j in range(0,m):
        
        if (j==0):
            phi_matrix_valtn_syn[i][j] = 1
        else:
            
            diff = np.subtract(x_valtn_syn[i] , mu_matrix[j])
            
            phi_matrix_valtn_syn[i][j] =  (np.exp(np.dot(-0.5 , np.dot(  np.dot(diff,var_invmat_train_syn),(np.transpose(diff))))))
  
phi_matrix_tst_syn = np.zeros((syn_val_size,m))
    
for i in range(0,syn_val_size):
    
    for j in range(0,m):
        
        if (j==0):
            phi_matrix_tst_syn[i][j] = 1
        else:
            
            diff = np.subtract(x_tst_syn[i] , mu_matrix[j])
            
            phi_matrix_tst_syn[i][j] =  (np.exp(np.dot(-0.5 , np.dot(  np.dot(diff,var_invmat_train_syn),(np.transpose(diff))))))
                      
    
err_w = []
sum_sq_syn = 0

for i in range(0,syn_val_size):
    sq_diff_syn = np.square(rel_valtn_syn[i] - np.dot(w_ml_st_syn , phi_matrix_train_syn[i]))
    
    sum_sq_syn += sq_diff_syn
    
    
err_dw_valtn_syn = np.dot(sum_sq_syn , 0.5)



err_rms_valtn_syn = np.sqrt(2*err_dw_valtn_syn/syn_val_size)
print("RMS error-validation error(CFS)")
print(err_rms_valtn_syn)

#testing

rel_valtn_sgd_syn = np.zeros(syn_val_size)

for i in range(syn_train_size,18000):
    rel_valtn_sgd_syn[i-syn_train_size] = out_syn[i]
  

err_w = []
sum_sq_syn = 0

for i in range(0,syn_val_size):
    sq_diff_syn = np.square(rel_valtn_syn[i] - np.dot(w_ml_st_syn , phi_matrix_tst_syn[i]))
    
    sum_sq_syn += sq_diff_syn
    
    
err_dw_valtn_syn = np.dot(sum_sq_syn , 0.5)



err_rms_valtn_syn = np.sqrt(2*err_dw_valtn_syn/syn_val_size)
print("RMS error-testing error(CFS)")
print(err_rms_valtn_syn)

#stochastic gradient descent(Synthetic dataset)


w_sgd_new = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]

w_sgd_iter = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]


lmda = lmda_ml_syn
eta = 0.001

for i in range(0, syn_train_size):
    delta_ED = np.dot(np.subtract( rel_train_syn[i] , np.dot(np.transpose(w_sgd_iter) , phi_matrix_train_syn[i])) , phi_matrix_train_syn[i])

    del_w = np.dot(eta, np.subtract(delta_ED , np.dot(lmda, w_sgd_iter)))
    w_sgd_new = w_sgd_iter + del_w 
    w_sgd_iter = w_sgd_new
print("weights(SGD)")
print(w_sgd_iter)
err_w_sgd = []
sum_sq_sgd = 0

for i in range(0,syn_train_size):
    sq_diff_sgd = np.square(rel_train_syn[i] - np.dot(w_sgd_new , phi_matrix_train_syn[i]))
    
    sum_sq_sgd += sq_diff_sgd
    
    
err_dw_sgd = np.dot(sum_sq_sgd , 0.5)

err_rms_train_sgd = np.sqrt(2*err_dw_sgd/syn_train_size)
print("RMS error - training error (SGD)")
print(err_rms_train_sgd)

#validation set for stochastic gradient descent


rel_valtn_sgd_syn = np.zeros(syn_val_size)

for i in range(syn_train_size,18000):
    rel_valtn_sgd_syn[i-syn_train_size] = out_syn[i]
  
x_valtn_sgd = np.zeros((syn_train_size,ft_len))
  
for i in range(syn_train_size,18000):
    x_valtn_sgd[i- syn_train_size] = in_syn[i]
    

    
err_w = []
sum_sq_sgd = 0

for i in range(0,syn_val_size):
    sq_diff_sgd = np.square(rel_valtn_sgd_syn[i] - np.dot(w_sgd_new , phi_matrix_valtn_syn[i]))
    
    sum_sq_sgd += sq_diff_sgd
    
    
err_dw_valtn_sgd = np.dot(sum_sq_sgd , 0.5)



err_rms_valtn_sgd = np.sqrt(2*err_dw_valtn_sgd/syn_val_size)
print("RMS error-validation error(SGD)")
print(err_rms_valtn_sgd)

#testing set for stochastic gradient descent

rel_valtn_sgd_syn = np.zeros(syn_tst_size)

for i in range(18000,20000):
    rel_valtn_sgd_syn[i-18000] = out_syn[i]
  
x_valtn_sgd = np.zeros((syn_train_size,ft_len))
  
for i in range(18000,20000):
    x_valtn_sgd[i- 18000] = in_syn[i]
    

    
err_w = []
sum_sq_sgd = 0

for i in range(0,syn_val_size):
    sq_diff_sgd = np.square(rel_valtn_sgd_syn[i] - np.dot(w_sgd_new , phi_matrix_tst_syn[i]))
    
    sum_sq_sgd += sq_diff_sgd
    
    
err_dw_valtn_sgd = np.dot(sum_sq_sgd , 0.5)



err_rms_valtn_sgd = np.sqrt(2*err_dw_valtn_sgd/syn_val_size)
print("RMS error-testing error(SGD)")
print(err_rms_valtn_sgd)

print("Hyperparameters for synthetic data")

print("M")
print(m)
print("lambda")
print(lmda)
print("eta")
print(eta)






