# the program allows the user to draw a rectangle 
# by dragging and dropping the mouse pointer

import cv2
import numpy as np

# Variables
drawing = False
ix, iy = -1, -1


# Function
def draw_rectangle(event, x, y, flags, parameters):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True,
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), thickness=-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), thickness=-1)


# Showing the image

# create black image, three channels
img = np.zeros((512, 512, 3))
cv2.namedWindow(winname='mydrawing')
cv2.setMouseCallback('mydrawing', draw_rectangle)

while True:

    # show the img inside of the window
    cv2.imshow('mydrawing', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
