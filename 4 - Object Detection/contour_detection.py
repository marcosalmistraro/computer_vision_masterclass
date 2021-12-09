# identifying internal and external contours in cv2

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('internal_external.png', 0)

contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_CCOMP,
                                       method=cv2.CHAIN_APPROX_SIMPLE)

# Create a totally black canvas for adding ext contours later
external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)

# External contours are indexed as -1 in the contours array
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)

# Internal contours are all indexed as != -1 
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)

plt.imshow(external_contours, cmap='gray')
plt.title("external_contours")
plt.show()
plt.imshow(internal_contours, cmap='gray')
plt.title('internal_contours')
plt.show()
