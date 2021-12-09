import cv2
import matplotlib.pyplot as plt


def display_img(img):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


gorilla = cv2.imread('gorilla.jpg', 0)
hist_values = cv2.calcHist([gorilla], channels=[0], mask=None,
                           histSize=[256], ranges=[0, 256])
plt.plot(hist_values)
plt.title('histogram for gorilla.jpg')
plt.show()

eq_gorilla = cv2.equalizeHist(gorilla)
display_img(eq_gorilla)  
hist_values_eq = cv2.calcHist([eq_gorilla], channels=[0], mask=None,
                              histSize=[256], ranges=[0, 256])
                              
# display the equalized histogram
plt.plot(hist_values_eq)
plt.title('equalized histogram for gorilla.jpg')
plt.show()  

# equalize a color picture
color_gorilla = cv2.imread('gorilla.jpg')
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
display_img(show_gorilla)

hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # substitute the equalized values for the value channel
eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
display_img(eq_color_gorilla)
