# Johnnyapu15, ~20190325
# Cox's method

from scipy import ndimage
from scipy import fftpack
from scipy import signal
import numpy as np 
import matplotlib.pyplot as plt

# FUNC. Create watermarks
def createWatermarks(_num, _length, _str):
    if _str == "uniform":
        return np.random.rand(_num, _length)
    elif _str == "normal":
        return np.random.normal(0, 1, (_num, _length))

# FUNC. Image to frequency domain
# input:
#  RGB image (shape: (x, y, 3))
def dct(_img):
    # It has 3-d image
    ret = np.zeros(_img.shape, np.float)
    for i in range(0, 3):
        # Its same with 'dct2()' of Matlab
        ret[..., i] = fftpack.dct(fftpack.dct(_img[..., i].T, norm='ortho').T, norm='ortho')
    return ret

# input:
#  RGB image (shape: (x, y, 3))
def idct(_dcted):
    # It has 3-d dct-ed image
    ret = np.zeros(_dcted.shape, 'uint8')
    for i in range(0, 3):
        ret[..., i] = fftpack.idct(fftpack.idct(_dcted[..., i].T, norm='ortho').T, norm='ortho')
    return ret

# FUNC. For frequency domain f, aquire 'perceptual mask' v
# Lowpass filter 1-colored 2d image using ranking
# input:
#  1-colored image _d
#  ranking number _n
def lf2d(_d, _n):
    vect = _d.flatten()
    shape = _d.shape
    idx = vect.argsort()[::-1][:_n]
    return idx, vect, shape


# FUNC. For a perceptual mask v, insert watermark x.
def insertWatermark2d(_d_vect, _idx, _wm, _alpha):
    ret = _d_vect.copy()
    for idx, i in enumerate(_idx):
        ret[i] = _d_vect[i] + _alpha * _wm[[idx]]
    return ret

# Insert watermark _wm into _img using _alpha for the strength of watermark
def insertWatermark(_img, _wm, _alpha):
    n = _wm.size
    dcted = dct(_img)
    ret = np.zeros(_img.shape, np.float)
    for i in range(0, 3):
        idx, vect, shape = lf2d(dcted[..., i], n)
        r = insertWatermark2d(vect, idx, _wm, _alpha)
        ret[..., i] = r.reshape(shape)
    ret = idct(ret)
    return ret
# FUNC. Get watermark with the original image D and the distortured image D*.
def extractWatermark2d(_, _2d_star, _n):
    
# FUNC. Calculate sim()


def test():
    file_route = "./img/twice_01.jpg"
    d = ndimage.imread(file_route)
    wm = np.random.normal(0, 1, 1000)
    alpha = 0.1
    print(wm)
    ret = insertWatermark(d, wm, alpha)
    
    plt.imshow(d)
    plt.show()
    plt.imshow(ret)
    plt.show()
    plt.imshow((d - ret) * 255)
    plt.show()

test()