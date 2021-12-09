import cv2
import matplotlib.pyplot as plt
import numpy as np

flat_chessboard = cv2.imread('flat_chessboard.png')
gray_flat_chess = cv2.cvtColor(flat_chessboard, cv2.COLOR_BGR2GRAY)
real_chess = cv2.imread('real_chessboard.jpg')
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

# Convert image to 32 floating point
gray_flat_chess = np.float32(gray_flat_chess)

# Apply algorithm
dst = cv2.cornerHarris(src=gray_flat_chess, blockSize=2, ksize=3,
                       k=0.04)
dst = cv2.dilate(dst, None)  # for better visualization

# Wherever the result of corner detection is greater than 1% of the original value,
# highlight in red
flat_chessboard[dst > 0.01 * dst.max()] = [255, 0, 0]
plt.imshow(flat_chessboard)
plt.title('corner detection for flat board')
plt.show()

# Another example

gray_real_chess = np.float32(gray_real_chess)
dst_real_chess = cv2.cornerHarris(gray_real_chess, 2, 3, 0.04)
dst = cv2.dilate(dst_real_chess, None)
real_chess[dst_real_chess > 0.01 * dst_real_chess.max()] = (255, 0, 0)
plt.imshow(real_chess)
plt.title('corner detection for real board')
plt.show()
