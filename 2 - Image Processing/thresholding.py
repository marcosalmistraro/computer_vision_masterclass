# showing implementations of different thresholding techniques in cv2

import cv2
import matplotlib.pyplot as plt


def show_pic(img):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img1 = cv2.imread('rainbow.jpg', 0)
ret1, thresh1 = cv2.threshold(src=img1, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(src=img1, thresh=127, maxval=255, type=cv2.THRESH_TRUNC)

img2 = cv2.imread('gorilla.jpg', 0)
ret3, thresh3 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(img2, 180, 255, cv2.THRESH_TRUNC)

thresh_ad_rainbow = cv2.adaptiveThreshold(img1, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                  thresholdType=cv2.THRESH_BINARY, blockSize=11, C=8)
thresh_ad_gorilla = cv2.adaptiveThreshold(img2, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                  thresholdType=cv2.THRESH_BINARY, blockSize=11, C=8)

adaptive_rainbow = cv2.addWeighted(src1=ret3, alpha=0.8, src2=thresh_ad_rainbow, beta=0.4, gamma=0)
show_pic(adaptive_rainbow)
thresh1_rainbow = cv2.addWeighted(src1=ret1, alpha=0.8, src2=thresh1, beta=0.4, gamma=0)
show_pic(thresh1_rainbow)
thresh2_rainbow = cv2.addWeighted(src1=ret2, alpha=0.8, src2=thresh2, beta=0.4, gamma=0)
show_pic(thresh2_rainbow)

adaptive_gorilla = cv2.addWeighted(src1=ret3, alpha=0.8, src2=thresh_ad_gorilla, beta=0.4, gamma=0)
show_pic(adaptive_gorilla)
thresh3_gorilla = cv2.addWeighted(src1=ret3, alpha=0.8, src2=thresh3, beta=0.4, gamma=0)
show_pic(thresh3_gorilla)
thresh4_gorilla = cv2.addWeighted(src1=ret4, alpha=0.8, src2=thresh4, beta=0.4, gamma=0)
show_pic(thresh4_gorilla)
