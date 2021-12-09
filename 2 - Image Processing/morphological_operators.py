import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_img():
    blank_image = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_image, text='Test', org=(50, 300), fontFace=font,
                fontScale=5, color=(255, 255, 255), thickness=5)
    return blank_image


def display_img(img):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img1 = load_img()

# Erosion example

kernel = np.ones((5, 5), dtype=np.uint8)
result = cv2.erode(img1, kernel, iterations=1)
display_img(result)
cv2.imwrite('erosion.jpg', result)

# opening and closing example with white and black noise

white_noise = np.random.randint(low=0, high=2, size=(600, 600))
white_noise *= 255
noise_img = white_noise + img1
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
display_img(opening)
cv2.imwrite('white_noise.jpg', opening)

black_noise = np.random.randint(low=0, high=2, size=(600, 600))
black_noise *= -255
black_noise_img = img1 + black_noise
black_noise_img[black_noise_img == -255] = 0
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
display_img(closing)
cv2.imwrite('black_noise.jpg', closing)

# gradient example - edge detection

gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)
cv2.imwrite('gradient.jpg', gradient)
