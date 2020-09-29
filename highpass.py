# -*- coding: utf-8 -*-
import sys 
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def highpass(img):
    #傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    #设置高通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg

imgs = sys.argv[1]
#imgs = 'n0153282900000317_adv_sp.jpg'
#img = Image.open('..\\gaussian_noise\\' + imgs)
#img = Image.open('..\\poisson_noise\\' + imgs)
img = Image.open('..\\s&p_noise\\' + imgs)
#img = Image.open('..\\multiplicative_noise\\' + imgs)
r,g,b=img.split()
#r1=highpass(np.array(r))
#print(r1)
r1=Image.fromarray(np.uint8(highpass(np.array(r))))
g1=Image.fromarray(np.uint8(highpass(np.array(r))))
b1=Image.fromarray(np.uint8(highpass(np.array(r))))
img = Image.merge('RGB',[r,g,b])
pic = Image.merge('RGB',[r1,g1,b1])   
##显示原始图像和高通滤波处理图像
#plt.imshow(img), plt.title('Original Image')
#plt.axis('off')
#plt.imshow(pic), plt.title('Original Image')
#plt.axis('off')
#plt.show()
pic.save('..\\highpass\\' + imgs[:-4] + '_highpass.jpg')