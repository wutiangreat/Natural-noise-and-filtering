import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util
import skimage.io
#imgs = sys.argv[1]
imgs = 'target_0_base_9_6158.png'
img = Image.open('D:\\YWX\\filter\\test\\' + imgs)
img = np.array(img)
#
##添加噪声
##noise_gs_img = Image.fromarray(np.uint8(255*(util.random_noise(img,mode='gaussian', clip=True)))).convert('RGB')             
##noise_ps_img = Image.fromarray(np.uint8(255*(util.random_noise(img,mode='poisson', clip=True)))).convert('RGB')
##noise_sp_img = Image.fromarray(np.uint8(255*(util.random_noise(img,mode='s&p')))).convert('RGB')
noise_speckle_img = util.random_noise(img,mode='speckle', clip=True)
##保存噪声图片
##noise_gs_img.save('..\\gaussian_noise\\' + imgs[:-4] + '_gs.jpg')
##noise_ps_img.save('..\\poisson_noise\\' + imgs[:-4] + '_ps.jpg')
##noise_sp_img.save(imgs[:-4] + '_sp.jpg')
#noise_speckle_img.save(imgs[:-4] + '_speckle.jpg')
skimage.io.imsave('D:\\YWX\\filter\\test\\target_0_base_9_6158_speckle.png',noise_speckle_img)
#plt.imshow(noise_speckle_img)
#plt.axis('off')
#plt.show()