# the program displays:
# - a yellow circle when the left mouse button is pressed
# - a light blue circle when the right mouse button is pressed

import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, center=(x, y), radius=100, color=(0, 150, 255), thickness=-1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, center=(x, y), radius=100, color=(150, 150, 0), thickness=-1)


cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)

img = np.zeros((512, 512, 3), dtype=np.int8)

while True:

    cv2.imshow('my_drawing', img)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()