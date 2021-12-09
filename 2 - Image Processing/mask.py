import cv2
import matplotlib.pyplot as plt
import numpy as np

rainbow = cv2.imread('rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

img = rainbow
print(rainbow.shape)
# create mask for img
mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
plt.imshow(mask, cmap='gray') # pure black mask in 2D with same dimensions as 'img'
plt.title('full black mask')
plt.show()
mask[300:400, 100:400] = 255 # resized mask to only leave one selected open window
plt.imshow(mask, cmap='gray')
plt.title('resized mask')
plt.show()

# create masked img by using bitwise_and
masked_img = cv2.bitwise_and(img, img, mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)
plt.imshow(show_masked_img)
plt.title('masked rainbow')
cv2.imwrite('masked_rainbow.jpg', masked_img)
plt.show()

hist_mask_values_red = cv2.calcHist([rainbow], [2], mask=mask,
                                    histSize=[256], ranges=[0, 256])
hist_values_red = cv2.calcHist([rainbow], [2], mask=None,
                               histSize=[256], ranges=[0, 256])

plt.plot(hist_mask_values_red)
plt.title('red Histogram for masked rainbow')
plt.show()

plt.plot(hist_values_red)
plt.title('red Histogram for unmasked rainbow')
plt.show()
