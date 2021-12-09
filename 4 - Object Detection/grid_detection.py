# identifying a grid in cv2

import cv2
import matplotlib.pyplot as plt

flat_chessboard = cv2.imread('flat_chessboard.png')
found, corners = cv2.findChessboardCorners(flat_chessboard, patternSize=(7, 7))
# found is a bool for stating if a chessboard has been identified

cv2.drawChessboardCorners(flat_chessboard, patternSize=(7, 7),
                          corners=corners, patternWasFound=found)
plt.imshow(flat_chessboard)
plt.show()

# example with a circle grid
dots = cv2.imread('dot_grid.png')
found, dots_corners = cv2.findCirclesGrid(dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)
cv2.drawChessboardCorners(dots, (10, 10), dots_corners, found)
plt.imshow(dots)
plt.show()
