import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fftpack
from skimage import img_as_float
from skimage import img_as_ubyte

def correlate(original, x):
    shape = np.array(original.shape)
    
    fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    
    sp1 = fftpack.fftn(original, fshape)
    sp2 = fftpack.fftn(x, fshape)
    ret = fftpack.ifftn(sp1 * sp2)[fslice].copy().real
    return ret

#自适应维纳滤波
# count 为窗口数，original 为原图
def adaptiveWienerDeNoise(count, original):
    count = np.asarray(count)
    x = np.ones(count)[[slice(None, None, -1)] * np.ones(count).ndim].conj()
    # Estimate the local mean
    lMean = correlate(original, x) / np.product(count, axis=0)

    # Estimate the local variance
    lVar = correlate(original ** 2, x) /np.product(count, axis=0) - lMean ** 2

    # Estimate the noise power if needed.
    noise = np.mean(np.ravel(lVar), axis=0)

    res = (original - lMean)
    res *= (1 - noise / lVar)
    res += lMean
    out = np.where(lVar < noise, lMean, res)

    return out

#imgs = sys.argv[1]
imgs = 'n0153282900000317_adv.jpg'
original = Image.open(imgs).convert('RGB')
r,g,b=original.split()
#rDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(r))))
#rDeNoise = adaptiveWienerDeNoise([5,5], img_as_float(r))
rDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(r))))
gDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(g))))
bDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(b))))
pic = Image.merge('RGB',[rDeNoise,gDeNoise,bDeNoise])  
#plt.imshow(original)
#plt.show()
plt.imshow(pic)
plt.show()
pic.save(imgs[:-4] + '_wiener2_2.png')