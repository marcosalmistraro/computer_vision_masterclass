import cv2
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


reeses = cv2.imread('reeses_puffs.png', 0)
cereals = cv2.imread('many_cereals.jpg', 0)

# create object on both target and analyzed img
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

# calculate the k best matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# if match1 distance is less than 75% of match2 distance
# then let's keep the descriptor match, since it was a good one
# this is called ratio test
good = []
for match1, match2 in matches:
    if match1.distance < 0.75 * match2.distance:
        good.append([match1])

sift_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=2)
display(sift_matches)
