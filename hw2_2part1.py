from skimage import io
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from pylab import *
from skimage import io, color
image = io.imread('findIt.jpg')
"""
_ = plt.hist(image.ravel(), bins = 8 )
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
"""

img = cv.imread('findIt.jpg', 0)
equ = cv.equalizeHist(img)
figure(0)
#plt.imshow(equ)
_ = plt.hist(equ.ravel(), bins = 8 )
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
threshold, thresh = cv.threshold(equ, 100, 150, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)
figure(1)
io.imshow(equ)

res = np.hstack((img, equ)) #stacking images side-by-side
cv.imwrite('res.png', res)
plt.show()
io.show()
""""
#_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color='red', alpha = 0.5)
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color='Green', alpha = 0.5)
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color='Blue', alpha = 0.5)
_ = plt.xlabel('Intensity Value')
_ = plt.ylabel('Count')
#_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()
Output: Figure-2 """