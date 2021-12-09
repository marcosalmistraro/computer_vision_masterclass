import matplotlib.pyplot as plt
import cv2

# read image using cv2
img = cv2.imread('Image 1.jpg')
print(type(img)) # print data type
print(img) # print array
print(img.shape) # print array shape

# change color channels for image 1
fix_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# generate gray-only image
img_gray = cv2.imread('Image 1.jpg', cv2.IMREAD_GRAYSCALE)
# it does not require 3 channels anymore
print(img_gray.shape)
print(img_gray.max())

# using plt to display images
plt.imshow(fix_image, cmap='gray')
plt.show()
plt.imshow(img_gray, cmap='gray')
plt.show()
