import cv2 as cv
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
from PIL import Image
from math import sqrt

img = io.imread('oldPhoto.jpg')
print(img.shape)
b, g, r = cv.split(img)

total = np.zeros((600, 457))
for i in range(600):
    for j in range(457):
        total[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
print(total.shape)
"""
df = pd.DataFrame(total.flatten())
filepath = 'pixel_valuestoplam.xlsx'
df.to_excel(filepath, index=False)
"""
print(total)
"""
df = pd.DataFrame(avg.flatten())
filepath = 'average.xlsx'
df.to_excel(filepath, index=False)
"""
"""
df = pd.DataFrame(img.flatten())
filepath = 'pixel_values.xlsx'
df.to_excel(filepath, index=False)
"""
figure(0)
plt.imshow((np.abs(img).astype(int)))
print(img.shape)

#img = color.rgb2gray(img)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

img_fftshift = np.fft.fftshift(np.fft.fft2(img))

blank = np.ones(img.shape[:2], dtype='uint8')*255
cv.circle(blank, (138, 70), 17, 0, -1)
cv.circle(blank, (319, 70), 17, 0, -1)
cv.circle(blank, (138, 530), 17, 0, -1)
cv.circle(blank, (319, 530), 17, 0, -1)
cv.circle(blank, (49, 300), 13, 0, -1)
cv.circle(blank, (404, 300), 13, 0, -1)
cv.imshow('Blank Image', blank)

plt.figure(num=None, figsize=(8, 6), dpi=80)
figure(1)
plt.imshow(np.log(1+np.abs(img_fftshift)), "gray"), plt.title("Shifted FFT")

img_fftshift = img_fftshift * blank
img_fftshift = img_fftshift/255
plt.figure(num=None, figsize=(8, 6), dpi=80)
figure(2)
plt.imshow(np.log(1+np.abs(img_fftshift)), "gray"), plt.title("Centered Spectrum")
img_ifft = np.fft.ifft2(np.fft.ifftshift(img_fftshift))
figure(3)
plt.imshow(np.abs(img_ifft), "gray"), plt.title("Reversed Image")
"""
df = pd.DataFrame((np.abs(img_ifft).astype(int)).flatten())
filepath = 'pixel_values2.xlsx'
df.to_excel(filepath, index=False)
"""
#cv.imshow('IFFT', (np.abs(img_ifft).astype(int)))

avgNew = (np.abs(img_ifft).astype(int))
avgNew = avgNew * 3
bNew = np.zeros((600, 457))
gNew = np.zeros((600, 457))
rNew = np.zeros((600, 457))
for i in range(600):
    for j in range(457):
        bNew[i][j] = ((int(b[i][j]) / total[i][j]) * avgNew[i][j]).astype(int)
        gNew[i][j] = ((int(g[i][j]) / total[i][j]) * avgNew[i][j]).astype(int)
        rNew[i][j] = ((int(r[i][j]) / total[i][j]) * avgNew[i][j]).astype(int)
print(bNew)
merged = cv.merge([bNew, gNew, rNew])
figure(4)
plt.imshow((np.abs(merged).astype(int)))
cv.imwrite('merged.jpg', merged)
img_float32 = np.float32(merged)
merged = cv.cvtColor(img_float32, cv.COLOR_RGB2GRAY)

merged_fftshift = np.fft.fftshift(np.fft.fft2(merged))
plt.figure(num=None, figsize=(8, 6), dpi=80)
figure(5)
plt.imshow(np.log(1+np.abs(merged_fftshift)), "gray"), plt.title("Result")

"""
df = pd.DataFrame(merged.flatten())
filepath = 'merged.xlsx'
df.to_excel(filepath, index=False)
"""
plt.show()
io.show()
cv.waitKey(0)