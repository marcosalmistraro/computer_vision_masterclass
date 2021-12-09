import cv2
import matplotlib.pyplot as plt
import numpy as np

flat_chessboard = cv2.imread('flat_chessboard.png')
flat_chessboard = cv2.cvtColor(flat_chessboard, cv2.COLOR_BGR2RGB)

real_chess = cv2.imread('real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chessboard, cv2.COLOR_BGR2GRAY)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(image=gray_flat_chess, maxCorners=64,
                                  qualityLevel=0.01, minDistance=10)
corners_real = cv2.goodFeaturesToTrack(image=gray_real_chess, maxCorners=80,
                                       qualityLevel=0.01, minDistance=10)

corners = np.int0(corners)  # convert to integers for drawing circles
corners_real = np.int0(corners_real)  # also convert to integers

for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chessboard, (x, y), 3, (255, 0, 0), -1)
plt.imshow(flat_chessboard)
plt.show()

for i in corners_real:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)
plt.imshow(real_chess)
plt.show()
