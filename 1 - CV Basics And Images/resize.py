import numpy as np
import matplotlib.pyplot as plt
import cv2

# load img 1 using imread
img = cv2.imread("Image 1.jpg")
# fix color channels and display it
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
plt.show()

# generate resized img
new_img = cv2.resize(fix_img, (1000, 400))
plt.imshow(new_img)
plt.show()

# generate resized img using given ratios
w_ratio = 0.3
h_ratio = 0.6
new_img_2 = cv2.resize(fix_img, (0, 0), fix_img, w_ratio, h_ratio)
plt.imshow(new_img_2)
plt.show()
print(new_img_2.shape)

# flip img
flipped_img = cv2.flip(fix_img, -1)
plt.imshow(flipped_img)
flipped_img = cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR)
plt.show()
# output img to folder
cv2.imwrite('generated_flipped_img.jpg', flipped_img)