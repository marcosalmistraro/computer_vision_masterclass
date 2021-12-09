import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img():
    img = cv2.imread('bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_font_img():
    img_font = load_img()
    font_type = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img_font, text='bricks', org=(10, 600), fontFace=font_type, fontScale=10,
                color=(255, 0, 0), thickness=4)
    return img_font


def display_img(img):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


img1 = load_img()

# elevating all pixels from the np array to the gamma-power
gamma = 0.2
power_img = np.power(img1, gamma)
display_img(power_img)

# imposing text onto loaded img
img2 = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img2, text='bricks', org=(10,600), fontFace=font, fontScale=10,
            color=(255, 0, 0), thickness=4)
display_img(img2)

# applying kernel filter to img
kernel = np.ones(shape=(5,5), dtype=np.float32)/25
dst = cv2.filter2D(img2, -1, kernel)
display_img(dst)

# applying blur kernel
img3 = create_font_img()
blurred = cv2.blur(img3, ksize=(5, 5))
display_img(blurred)

# applying gaussian blur kernel
img4 = create_font_img()
blurred_gau = cv2.GaussianBlur(img4, ksize=(5, 5), sigmaX=10)
display_img(blurred_gau)

# applying median blur kernel
img5 = create_font_img()
blurred_med = cv2.medianBlur(img5, 5)
display_img(blurred_med)

# applying bilateral filter 
img6 = create_font_img()
img6 = cv2.bilateralFilter(img6, d=9, sigmaColor=75, sigmaSpace=75)
display_img(img6)
