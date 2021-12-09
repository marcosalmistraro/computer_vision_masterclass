import cv2
import matplotlib.pyplot as plt

gorilla = cv2.imread('gorilla.jpg')  # Original BRG channelling for OpenCV
show_gorilla = cv2.cvtColor(gorilla, cv2.COLOR_BGR2RGB)  # Converted to RGB for display

rainbow = cv2.imread('rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

hist_values_gorilla = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256],
                                 ranges=[0, 256])
hist_values_rainbow = cv2.calcHist([rainbow], channels=[0], mask=None, histSize=[256],
                                  ranges=[0, 256])

print(hist_values_rainbow.shape)
plt.plot(hist_values_rainbow)
plt.show()
plt.plot(hist_values_gorilla)
plt.show()

img = gorilla
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for gorilla img')
plt.show()

img = rainbow
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for rainbow img')
plt.show()
