import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, center=(x, y), radius=100, color=(0, 150, 255), thickness=-1)

# a window is created on top of which the img is displayed
# the MouseCallback action acts on the window
cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)

img = np.zeros((512, 512, 3), dtype=np.int8)

# standard loop for displaying and then destroying img when esc is pressed
while True:

    cv2.imshow('my_drawing', img)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
