import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util

# count 为最大窗口数，original 为原图
def KNN(k, count, original):
    # 卷积范围
    c = int(count/2)
    rows, cols = original.shape
    newI = original#zero matrix to store new picture
    for i in range(c, rows - c):
        for j in range(c, cols - c):
            distance_main = 0
            distance = []
            for x in range (i - c,i + c + 1) :
                for y in range (j - c,j + c + 1) :
                    if x != i and y != j:
                        distance.append(abs(original[i][j] - original[x][y]))
            distance.sort()
            for w in range(0,k):
                distance_main = distance_main + distance[w]
            distance_w = distance_main / w
            newI[i][j] =int(original[i][j] - distance_w ** 0.5)
    return newI

imgs = sys.argv[1]
#imgs = 'n0153282900000317_adv_gs.jpg'
#original = Image.open('..\\gaussian_noise\\' + imgs)
#original = Image.open('..\\poisson_noise\\' + imgs)
original = Image.open('..\\s&p_noise\\' + imgs)
#original = Image.open('..\\multiplicative_noise\\' + imgs)
r,g,b=original.split()
#adapMedianDeNoise = adaptiveMedianDeNoise(7, original)
rDeNoise = Image.fromarray(np.uint8(KNN(5,7, np.array(r))))
gDeNoise = Image.fromarray(np.uint8(KNN(5,7, np.array(g))))
bDeNoise = Image.fromarray(np.uint8(KNN(5,7, np.array(b))))
pic = Image.merge('RGB',[rDeNoise,gDeNoise,bDeNoise])  
#plt.imshow(pic)
#plt.axis('off')
#plt.show()
#plt.imshow(adapMedianDeNoise)
pic.save('..\\KNN\\' + imgs[:-4] + '_KNN.jpg')