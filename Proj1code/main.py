# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 00:38:49 2016

@author: arunchandrapendyala
"""

import xlrd
import numpy
from math import e, pi, sqrt , log
#import matplotlib.pyplot as plt

#SUBMITTED BY:

print "UBitName = apendyal"
print "personNumber = 50207136"



workbook = xlrd.open_workbook('univ_data.xls')
worksheet = workbook.sheet_by_index(0)

cs_scr=[]
r_ovr=[]
ad_pay=[]
tuit=[]

N = 49 # no. of data points in each variable set

for i in range(1,N+1):
    val1=worksheet.cell(i,2).value
    cs_scr.append(val1)
    val2=worksheet.cell(i,3).value
    r_ovr.append(val2)
    val3=worksheet.cell(i,4).value
    ad_pay.append(val3)   
    val4=worksheet.cell(i,5).value
    tuit.append(val4)
    
    #code for TASK 1
mu1 = numpy.mean(cs_scr)
print "mu1 = %0.2f " %mu1
mu2 = numpy.mean(r_ovr)
print "mu2 = %0.2f " %mu2
mu3 = numpy.mean(ad_pay)
print "mu3 = %0.2f " %mu3
mu4 = numpy.mean(tuit)
print"mu4 = %0.2f " %mu4


var1  = numpy.var(cs_scr)
print "var1 = %0.2f" %var1
var2  = numpy.var(r_ovr)
print "var2 = %0.2f" %var2
var3  = numpy.var(ad_pay)
print "var3 = %0.2f" %var3
var4  = numpy.var(tuit)
print "var4 = %0.2f" %var4

sigma1  = numpy.std(cs_scr)
print "sigma1 = %0.2f " %sigma1
sigma2  = numpy.std(r_ovr)
print "sigma2 = %0.2f " %sigma2
sigma3  = numpy.std(ad_pay)
print "sigma3 = %0.2f " %sigma3
sigma4 = numpy.std(tuit)
print "sigma4 = %0.2f " %sigma4

    #code for TASK 2

print "covarianceMat ="
cov = numpy.cov([cs_scr, r_ovr, ad_pay, tuit])
print numpy.around(cov,2)
print "correlationMat ="
corr = numpy.corrcoef([cs_scr, r_ovr, ad_pay, tuit])
print numpy.around(corr,2)

    #scatter plots for pairwise data sets
"""
plt.figure(1)
fig1=plt.scatter(cs_scr, r_ovr)
plt.xlabel("cs score")
plt.ylabel("research overhead")
plt.title("plot of cs score vs research overhead")
plt.show()

plt.figure(2)
plt.scatter(r_ovr, ad_pay)
plt.xlabel("research overhead")
plt.ylabel("administrative pay")
plt.title("plot of research overhead vs administrative pay")
plt.show()

plt.figure(3)
plt.scatter(r_ovr, tuit)
plt.xlabel("research overhead")
plt.ylabel("tuition")
plt.title("plot of research overhead vs tuition")
plt.show()

plt.figure(4)
plt.scatter(tuit, cs_scr)
plt.xlabel("tuition")
plt.ylabel("cs score")
plt.title("plot of tuition vs cs score")
plt.show()

plt.figure(5)
plt.scatter(cs_scr, ad_pay)
plt.xlabel("cs score")
plt.ylabel("administrative pay")
plt.title("plot of cs score vs administrative pay")
plt.show()

plt.figure(6)
plt.scatter(cs_scr, tuit)
plt.xlabel("cs score")
plt.ylabel("tuition")
plt.title("plot of cs score vs tuition")
plt.show()
"""
    #code for TASK 3-Normal distribution is assumed

def gauss(mu,var,val):
    fx = (1/sqrt(2*pi*var))*e**(-0.5*(float(val-mu)/sqrt(var))**2)
    return fx

sum =0

l1=l2=l3=l4 = 0

for i in range(0,N):
    l1 = l1 + log(gauss(mu1,var1,cs_scr[i])) 
    l2 = l2 + log(gauss(mu2,var2,r_ovr[i]))
    l3 = l3 + log(gauss(mu3,var3,ad_pay[i]))
    l4 = l4 + log(gauss(mu4,var4,tuit[i]))
    


logLikelihood =l1+ l2 + l3 + l4
print "logLikelihood = %0.2f" %logLikelihood
    #code for TASK 4


BNgraph = [[0, 0 ,0, 0],[1, 0, 0, 0],[1, 0 ,0 ,0],[1, 0, 0, 0]]

print "BNgraph =  " 
print(numpy.matrix(BNgraph))

x_0 = numpy.ones(N)
data_mat = numpy.column_stack((x_0,r_ovr,ad_pay,tuit)) # data set
y= numpy.transpose(cs_scr)
#finding beta values for y=x1
a_mat=numpy.zeros(shape=(4,4))

for i in range(0,4):
    for j in range(0,4):
        sum = 0
        for k in range(0,N):
            sum+=data_mat[k,i]*data_mat[k,j]
            a_mat[i,j]=sum
            

y_mat = numpy.zeros(shape=(4,1))

for i in range(0,4):
    sum=0
    for j in range(0,N):
        sum+=data_mat[j,i]*y[j] #choose  y!!!!
        y_mat[i,0]=sum
        
b_mat=numpy.linalg.solve(a_mat,y_mat)


s_bn=0
for i in range(0,N):
    sum2=0
    for j in range(0,4):
        sum2+= b_mat[j]*data_mat[i,j]
    sum2 -= y[i]
    s_bn += sum2*sum2
    
bn_var=(float(s_bn)/N)


#l_theta

c_term= -0.5*N*log(2*pi*bn_var)
term2 = -0.5 *float((s_bn)/bn_var)

l_theta= c_term + term2


BNLoglikelihood = l_theta + l2 + l3 +l4 #log terms of p(x1/x2,x3,x4) , p(x2), p(x3) , p(x4)

print("BNlogLikelihood = %0.2f") %BNLoglikelihood