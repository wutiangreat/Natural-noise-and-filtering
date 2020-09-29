# -*- coding: utf-8 -*-
import sys
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util

def changeNumpyToList(A, length ,weight):
    B = []
    for i in range(0, length):
        for j in range(0, weight):
            B.append(A[i][j])
    return B

def changeListToNumpy(A, length ,weight):
    k = 0
    B = [[0 for i in range(length)] for i in range(weight)]
    for i in range(0, length):
        for j in range(0, weight):
            B[i][j] = A[k]
            k = k + 1
    return B

#定义向量的内积
def multiVector(A,B): 
#    print(A)
    C=sc.zeros(len(A))    
    for i in range(len(A)):        
        C[i]=A[i]*B[i]  
#        print(C[i])
    return sum(C)

#取定给定的反向的个数
def inVector(A,b,a):
    D=sc.zeros(b-a+1)
    for i in range(b-a+1):
        D[i]=A[i+a]
    return D[::-1]

#lMS算法的函数
def LMS(xn,M,mu,itr):
    en=sc.zeros(itr)
    W=[[0]*M for i in range(itr)]  
    for k in range(itr)[M-1:itr]:
        x=inVector(xn,k,k-M+1)
        d= x.mean()
        y=multiVector(W[k-1],x)  
        en[k]=d -y      
        print(en[k])
        W[k]=np.add(W[k-1],2*mu*en[k]*x) #更新权重  
        print(W[k-1])
#        print(W[k])
    #求最优时滤波器的输出序列    
    yn=sc.inf*sc.ones(len(xn))    
    for k in range(len(xn))[M-1:len(xn)]:    
        x=inVector(xn,k,k-M+1)        
        yn[k]=multiVector(W[len(W)-1],x)    
    return (yn,en)

#参数设置
M=9 #滤波器的阶数    
mu=0.000001 #步长因子  
imgs = 'n0153282900000317_adv.png'
#imgs=sys.argv[1]
original = Image.open('..\\FGSM\\' + imgs)
r, g, b, a  = original.split()
length_r ,weight_r = np.array(r).shape
r = changeNumpyToList(np.array(r),length_r ,weight_r)
(yr,er)=LMS(r,M,mu,len(r)) 
yr = Image.fromarray(np.uint8(np.array(changeListToNumpy(yr, length_r ,weight_r))))

length_g ,weight_g = np.array(g).shape
g = changeNumpyToList(np.array(g),length_g ,weight_g)
(yg,eg)=LMS(g,M,mu,len(g)) 
yg = Image.fromarray(np.uint8(np.array(changeListToNumpy(yg, length_g ,weight_g))))

length_b ,weight_b = np.array(b).shape
b = changeNumpyToList(np.array(b),length_b ,weight_b)
(yb,eb)=LMS(b,M,mu,len(b)) 
yb = Image.fromarray(np.uint8(np.array(changeListToNumpy(yb, length_b ,weight_b))))

#pic = Image.merge('RGB',[yr,yg,yb]) 
#
#pic.save('..\\LMS\\' + imgs[:-4] + '_LMS.jpg')
plt.imshow(yr), plt.title('Original Image')
plt.axis('off')
plt.show()