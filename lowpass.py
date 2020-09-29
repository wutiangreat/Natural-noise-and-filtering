# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def lowpass(img):
    #傅里叶变换
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)
    
    #设置低通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    
    #掩膜图像和频谱图像乘积
    f = fshift * mask
    
    #傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
    return res

imgs = sys.argv[1]
#imgs = 'n01514859_4702_adv_sp.jpg'
img = Image.open('..\\gaussian_noise\\' + imgs)
#img = Image.open('..\\poisson_noise\\' + imgs)
#img = Image.open('..\\s&p_noise\\' + imgs)
#img = Image.open('..\\multiplicative_noise\\' + imgs)
r,g,b=img.split()
r1=Image.fromarray(np.uint8(lowpass(np.array(r))/100000-3))
g1=Image.fromarray(np.uint8(lowpass(np.array(g))/100000-3))
b1=Image.fromarray(np.uint8(lowpass(np.array(b))/100000-3)) 
pic = Image.merge('RGB',[r1,g1,b1])   
##显示原始图像和高通滤波处理图像
#plt.imshow(pic), plt.title('Original Image')
#plt.axis('off')
#plt.show()
pic.save('..\\lowpass\\' + imgs[:-4] + '_lowpass.jpg')