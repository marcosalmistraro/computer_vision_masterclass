# the following program identifies russian car plates using a dedicated haar cascade.
# it then automatically blurs the plate and outputs the resulting image

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display(img):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


def detect_plate(img):
    plate_img = img.copy()
    plate_rect = plate_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in plate_rect:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return plate_img


def detect_and_blur_plate(img):
    plate_img = img.copy()
    plate_rect = plate_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in plate_rect:
        roi = plate_img[y:y+h, x:x+w, :]
        roi = cv2.medianBlur(roi, 25)
        plate_img[y:y+h, x:x+w, :] = roi
    return plate_img


car_img = cv2.imread('car_plate.jpg')

plate_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

result = detect_plate(car_img)
display(result)
result2 = detect_and_blur_plate(car_img)
display(result2)
