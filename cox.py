# Johnnyapu15, ~20190325
# Cox's method

from scipy import ndimage
from scipy import misc
from scipy import fftpack
import numpy as np 
import matplotlib.pyplot as plt
import os
from datetime import datetime

# FUNC. Create watermarks
def createWatermarks(_num, _length, _str = 'normal'):
    ret = 0
    if _str == "uniform":
        ret = np.random.rand(_num, _length)
    elif _str == "normal":
        ret = np.random.normal(0, 1, (_num, _length))
    directory = './watermarks/' + str(int(datetime.now().timestamp() * 10))
    try:
        if not(os.path.isdir(directory)):
            os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!")
            raise
    #path = "./" + directory
    for i in range(_num):
        filename = directory + '/' + str(i) + '.txt'
        if os.path.isfile(filename):
            print("Error: duplicated file name of watermark.")
            pass
        else:
            file = open(filename, "w")
            st = str(ret[i])
            st = st.replace('[', '')
            st = st.replace(']', '')
            file.write(st)
            file.close()
    return ret, directory, _num

# Loading watermark data
def loadWatermark(_dir, num):
    ret = []
    for i in range(num):
        file = open(_dir + '/' + str(i) + '.txt', "r")
        ret.append(np.fromstring(str(file.read()), dtype=np.float, sep=' '))
        file.close()
    return np.asarray(ret)

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
    ret = np.ndarray.astype(ret, np.uint8)
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

# Generate noise
# @ self
# @ type (String): Noise type
# @ file (String): Image file
def generateNoise(_img, _arg, _type = "gaussian"):
    img_type = type(_img[0][0][0])
    range_mx = 0
    if np.float64 == img_type:
        range_mx = 1
    elif np.uint8 == img_type:
        range_mx = 255
    if _type == "gaussian":
        row, column, channel = _img.shape
        mean = _arg[0]
        sigma = _arg[1]
        gaussian = np.random.normal(mean, sigma, (row, column, channel))
        img_noised = _img + gaussian * range_mx
        return img_noised.astype(img_type) 
    elif _type == "poisson":
        pass


def show_cox(_img, _img_w, _wms, _alpha):
    sims = calcSims(_img, _img_w, _wms, _alpha)
    fig = plt.figure()
    fig.add_subplot(1, 3, 1).imshow(_img)
    fig.add_subplot(1, 3, 2).imshow(_img_w)
    fig.add_subplot(1, 3, 3).plot(sims)
    plt.show()

def show_coef(_coefs):
    fig = plt.figure()
    _coefs[0] = np.abs(_coefs[0]) + 1
    _coefs[1] = np.abs(_coefs[1]) + 1
    _coefs[2] = np.abs(_coefs[2]) + 1
    fig.add_subplot(1, 3, 1).imshow(fftpack.fftshift(np.log(_coefs[0])), cmap='gray')
    fig.add_subplot(1, 3, 2).imshow(fftpack.fftshift(np.log(_coefs[1])), cmap='gray')
    fig.add_subplot(1, 3, 3).imshow(fftpack.fftshift(np.log(_coefs[2])), cmap='gray')
    plt.show()

#file_route = "./img/lena.png"
# file_route = "./img/twice_01.jpg"


# alpha = 0.1
# num = 200
# length = 100

# d = ndimage.imread(file_route)
# wms, directory, _ = createWatermarks(num, length, 'normal')
# idx = 5
# print('Inserting watermark[' + str(idx) + ']...')
# ret, coefs = insertWatermark(d, wms[idx], alpha)
# coefs[0] = np.abs(coefs[0]) + 1
# coefs[1] = np.abs(coefs[1]) + 1
# coefs[2] = np.abs(coefs[2]) + 1

# sims = calcSims(d, ret, wms, alpha)
# fig = plt.figure()
# fig.add_subplot(2, 3, 1).imshow(d)
# fig.add_subplot(2, 3, 2).imshow(ret)
# fig.add_subplot(2, 3, 3).plot(sims)
# fig.add_subplot(2, 3, 4).imshow(fftpack.fftshift(np.log(coefs[0])), cmap='gray')
# fig.add_subplot(2, 3, 5).imshow(fftpack.fftshift(np.log(coefs[1])), cmap='gray')
# fig.add_subplot(2, 3, 6).imshow(fftpack.fftshift(np.log(coefs[2])), cmap='gray')
# plt.show()


# Set variables
n = 200
l = 1000
a = 0.1
#r = "./img/lena.png"
r = './img/twice_01.jpg'
s = './img-w/'
print('n: the number of watermarks')
print('l: the length of watermark')
print('a: the alpha in algorithm')
print('r: the route of image')


wm_idx = 428

# Create watermarks
wms, directory, _ = createWatermarks(n, l)

# Load watermarks
loaded = loadWatermark(directory, n)

# Load image
img = ndimage.imread(r)

# Insert a watermark into image
img_w, coefs = insertWatermark(img, wms[wm_idx], a)

# Save watermarked-image
misc.imsave(s + directory.split('/')[-1] + '-' + str(wm_idx) + '.jpg', img_w)

# Plot 
# show_cox(img, img_w, wms, a)
# show_coef(coefs)

# Get noised image
noised = generateNoise(img, (0, 0.01)) # Gaussian random noise


# Experiment using gaussian noised image
noised_arr = []
fig = plt.figure()
test_case = 5
for i in range(test_case):
    noised_arr.append(generateNoise(img_w, (0, 0.001 * (i))))
    sims = calcSims(img, noised_arr[-1], wms, a)
    fig.add_subplot(test_case, 3, 1 + i * 3).imshow(img_w)
    fig.add_subplot(test_case, 3, 2 + i * 3).imshow(noised_arr[-1])
    fig.add_subplot(test_case, 3, 3 + i * 3).plot(sims)

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
