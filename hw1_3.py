from skimage import io, color, data
import pandas as pd
from pylab import *

img = io.imread('resim3.jpg')
print(img.shape)


df = pd.DataFrame(img.flatten())
filepath = 'pixel_values.xlsx'
df.to_excel(filepath, index=False)

# io.imshow(img)  # Displays an image.

# Covert to HSV
img_hsv = color.rgb2hsv(img)
# Covert back to RGB
img_rgb = color.hsv2rgb(img_hsv)

# Show both figures
figure(0)
io.imshow(img_hsv)
figure(1)
io.imshow(img_rgb)

io.show()

