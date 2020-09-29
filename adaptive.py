import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util

def adaptiveMedianDeNoise(count, original):
    startWindow = 3
    c = int(count/2)
    rows, cols = original.shape
    newI = original#zero matrix to store new picture
    for i in range(c, rows - c):
        for j in range(c, cols - c):
            k = int(startWindow / 2)
            median = np.median(original[i - k:i + k + 1, j - k:j + k + 1])
#            print(original)#test
#            print("i-k=",i-k);
#            print("i+K+1=",i+k+1);
#            print("j-k=",j-k);
#            print("j+k+1=",j+k+1);
#            print(original[i - k:i + k + 1, j - k:j + k + 1])
            mI = np.min(original[i - k:i + k + 1, j - k:j + k + 1])
            ma = np.max(original[i - k:i + k + 1, j - k:j + k + 1]) 
            if mI < median < ma:
                if mI < original[i, j] < ma:
                    newI[i, j] = original[i, j]
                else:
                    newI[i, j] = median

            else:
                while True:
                    if mI < median < ma or startWindow > count or startWindow == count:
                        break
                    startWindow = startWindow + 2
                    k = int(startWindow / 2)
                    median = np.median(original[i - k:i + k + 1, j - k:j + k + 1])
                    print('mI=',int(np.min(original[i - k:i + k + 1, j - k:j + k + 1])))
                    print('i-k:i+k',i-k,':',i+k+1)
                    print('j-k:j+k+1',j-k,':',i+k+1)
                    print('original',original)
                    print('original_shape',original.shape)
                    mI = np.min(original[i - k:i + k + 1, j - k:j + k + 1])
                    ma = np.max(original[i - k:i + k + 1, j - k:j + k + 1])
                    if mI < median < ma or startWindow > count or startWindow == count:
                        break
                if mI < median < ma or startWindow > count or startWindow == count:
                    if mI < original[i, j] < ma:
                        newI[i, j] = original[i, j]
                    else:
                        newI[i, j] = median

    return newI

imgs = sys.argv[1]
#imgs = 'n01514859_4702_adv_gs.jpg'
original = Image.open('..\\gaussian_noise\\' + imgs)
#original = Image.open('..\\poisson_noise\\' + imgs)
#original = Image.open('..\\s&p_noise\\' + imgs)
#original = Image.open('..\\multiplicative_noise\\' + imgs)
r,g,b=original.split()
adapMedianDeNoise = adaptiveMedianDeNoise(7, original)
rDeNoise = Image.fromarray(np.uint8(adaptiveMedianDeNoise(7, np.array(r))))
gDeNoise = Image.fromarray(np.uint8(adaptiveMedianDeNoise(7, np.array(g))))
bDeNoise = Image.fromarray(np.uint8(adaptiveMedianDeNoise(7, np.array(b))))
pic = Image.merge('RGB',[rDeNoise,gDeNoise,bDeNoise])  
plt.imshow(pic), plt.title('Original Image')
plt.axis('off')
plt.show()
#plt.imshow(adapMedianDeNoise)
plt.show()
pic.save('..\\adaptive\\' + imgs[:-4] + '_adaptive.jpg')