import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io

imgs = sys.argv[1]
#imgs = '001.png'
img = cv2.imread('D:\\YWX\\filter\\test\\' + imgs,0)
img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_Guassian = cv2.GaussianBlur(img,(5,5),0)
img_median = cv2.medianBlur(img,5)
img_mean = cv2.blur(img, (3,3))
img_bilater = cv2.bilateralFilter(img,9,75,75)

img_Guassian = Image.fromarray(np.uint8(img_Guassian))
img_median = Image.fromarray(np.uint8(img_median))
img_mean = Image.fromarray(np.uint8(img_mean))
img_bilater = Image.fromarray(np.uint8(img_bilater)

img_Guassian.save('..\\guassian\\' + imgs[:-4] + '_guassian.jpg')
img_median.save(imgs[:-4] + '_median.jpg')
img_mean.save(imgs[:-4] + '_mean.jpg')
img_bilater.save('..\\bilater\\' + imgs[:-4] + '_bilater.jpg')
img_Guassian.save("strawberry_guassian.jpg")
img_median.save("strawberry_median.jpg")
img_mean.save("strawberry_mean.jpg")
img_bilater.save("strawberry_bilater.jpg")


titles = ['srcImg','mean', 'Gaussian', 'median', 'bilateral']
imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()
plt.imshow(img_median)
plt.axis('off')
plt.show()
skimage.io.imsave('D:\\YWX\\filter\\test\\002.png',img_median)