import matplotlib.pyplot as plt
from skimage import io, color, data, exposure, transform
from pylab import *
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image, ImageChops
import PIL

img = io.imread('findIt.jpg')
figure(0)
plt.imshow(img)
img = 255-img
figure(1)
plt.imshow(img)
img2_rgb2gray = color.rgb2gray(img)
fft_image = np.fft.fftshift(np.fft.fft2(img2_rgb2gray))
plt.figure(num=None, figsize=(8, 6), dpi=80)
figure(2)
plt.imshow(np.log(abs(fft_image)), cmap='gray')  # Displays an image.    <1>

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)
figure(3)
plt.imshow((np.abs(thresh).astype(int)))
# Simple Thresholding Inverse
threshold_inv, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse', thresh_inv)
figure(4)
plt.imshow((np.abs(thresh_inv).astype(int)))
torpak = thresh + thresh_inv
figure(5)
plt.imshow((np.abs(torpak).astype(int)))
# Adaptive Thresholding
#adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
#cv.imshow('Adaptive Thresholded', adaptive_thresh)

# Adaptive Thresholding Inverse
#adaptive_thresh_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 3)
#cv.imshow('Adaptive Thresholded Inverse', adaptive_thresh_inv)

cv.waitKey(0)
plt.show()