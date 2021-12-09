import cv2
import numpy as np
import matplotlib.pyplot as plt

# initialize canvas
img = np.zeros(shape=(512, 512, 3), dtype=np.int32)

# put text on canvas
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, text='Hello', org=(10, 500), fontFace=font, fontScale=4, color=(255, 255, 255),
            thickness=3, lineType=cv2.LINE_AA)

# draw polygon - vertices vector has to be reshaped
vertices = np.array([[100, 300], [200, 200], [400, 300], [200, 400]], dtype=np.int32)
points = vertices.reshape((-1, 1, 2))
cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)

print(img.shape)
plt.imshow(img)
plt.show()
