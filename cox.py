# Johnnyapu15, ~20190325
# Cox's method

from scipy import ndimage
from scipy import fftpack
import numpy as np 
import matplotlib.pyplot as plt
import os

# FUNC. Create watermarks
def createWatermarks(_num, _length, _str = 'normal'):
    ret = 0
    if _str == "uniform":
        ret = np.random.rand(_num, _length)
    elif _str == "normal":
        ret = np.random.normal(0, 1, (_num, _length))
    path = ''
    for i in range(_num):
        if os.path.isfile("./watermarks/.txt"):
            pass
        else:
            file = open("./watermarks/watermark.txt", "w")
            file.write(str(watermark))
            file.close()
def generateWatermark(size):
        size = size 
        np.random.seed(100)
        watermark = np.random.normal(size=size)
        

# Loading watermark data
def loadWatermark():
    file = open("./data/watermark.txt", "r")
    watermark = file.read()
    file.close()
    return watermark
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
    ret = np.zeros(_dcted.shape, np.float)
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
def insertWatermark2d(_d_vect, _idx, _wm, _alpha, _type=1):
    ret = _d_vect.copy()
    if _type == 1:
        for idx, i in enumerate(_idx):
            ret[i] = _d_vect[i] + _alpha * _wm[[idx]]
    return ret

# Insert watermark _wm into _img using _alpha for the strength of watermark
def insertWatermark(_img, _wm, _alpha):
    n = _wm.size
    img_0_1 = _img / 255
    dcted = dct(img_0_1)
    ret = np.zeros(img_0_1.shape, np.float)
    ret_coef = []
    for i in range(0, 3):
        idx, vect, shape = lf2d(dcted[..., i], n)
        r = insertWatermark2d(vect, idx, _wm, _alpha)
        ret[..., i] = r.reshape(shape)
        ret_coef.append(dcted[..., i])
    ret = idct(ret)
    ret *= 255
    ret = np.ndarray.astype(ret, np.int)
    return ret, ret_coef
# FUNC. Get watermark with the original image D and the distortured image D*.
# 1-colored
def extractWatermark2d(_d, _d_star, _n, _alpha, _type = 1):
    dcted_d = fftpack.dct(fftpack.dct(_d.T, norm='ortho').T, norm='ortho')
    dcted_d_star = fftpack.dct(fftpack.dct(_d_star.T, norm='ortho').T, norm='ortho')
    dd_flat = (dcted_d / 255).flatten()
    dd_star_flat = (dcted_d_star / 255).flatten()
    v_idx, _, _ = lf2d(dcted_d, _n)  # v_idx: perceptual mask
    tmp2 = np.zeros(_n, np.float)
    if _type == 1: 
        for idx, j in enumerate(v_idx):
            tmp2[idx] = (dd_star_flat[j] - dd_flat[j]) / _alpha    
    return tmp2

def extractWatermark(_d, _d_star, _n, _alpha, _type = 1):
    ret = []
    for i in range(0, 3):
        ret.append(extractWatermark2d(_d[..., i], _d_star[..., i], _n, _alpha, _type))
    return ret

# FUNC. Calculate sim()
def sim(_x, _x_star):
    return (np.dot(_x, _x_star)) / np.sqrt(np.dot(_x_star, _x_star))

def calcSims(_d, _d_star, _wms, _alpha, _type = 1):
    n = _wms.shape
    # n(0): the number of watermarks
    # n(1): the length of watermark
    x_star = extractWatermark(_d, _d_star, n[1], _alpha, _type)
    sims = []
    for i in range(n[0]):
        sims.append(sim(_wms[i], x_star[0]))
    return sims




#file_route = "./img/lena.png"
file_route = "./img/twice_01.jpg"


alpha = 0.1
num = 200
length = 100

d = ndimage.imread(file_route)
wms = createWatermarks(num, length, 'normal')
idx = 5
print('Inserting watermark[' + str(idx) + ']...')
ret, coefs = insertWatermark(d, wms[idx], alpha)
coefs[0] = np.abs(coefs[0]) + 1
coefs[1] = np.abs(coefs[1]) + 1
coefs[2] = np.abs(coefs[2]) + 1

sims = calcSims(d, ret, wms, alpha)
fig = plt.figure()
fig.add_subplot(2, 3, 1).imshow(d)
fig.add_subplot(2, 3, 2).imshow(ret)
fig.add_subplot(2, 3, 3).plot(sims)
fig.add_subplot(2, 3, 4).imshow(fftpack.fftshift(np.log(coefs[0])), cmap='gray')
fig.add_subplot(2, 3, 5).imshow(fftpack.fftshift(np.log(coefs[1])), cmap='gray')
fig.add_subplot(2, 3, 6).imshow(fftpack.fftshift(np.log(coefs[2])), cmap='gray')
plt.show()


# # code for demo
# print('n: the number of watermarks')
# print('l: the length of watermark')
# print('a: the alpha in algorithm')
# print('r: the route of image')
# n = 200
# l = 100
# a = 0.1
# r = './img/twice_01.jpg'
