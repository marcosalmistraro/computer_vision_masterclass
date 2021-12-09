# showing different ways of encoding color values
# in NP arrays

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('rainbow.jpg')

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

plt.imshow(imgRGB)
plt.show()
plt.imshow(imgHSV)
plt.show()
plt.imshow(imgHLS)
plt.show()

imgHSV = cv2.cvtColor(imgHSV, cv2.COLOR_BGR2RGB)
imgHLS = cv2.cvtColor(imgHLS, cv2.COLOR_BGR2RGB)
cv2.imwrite('rainbowHSV.jpg', imgHSV)
cv2.imwrite('rainbowHLS.jpg', imgHLS)
