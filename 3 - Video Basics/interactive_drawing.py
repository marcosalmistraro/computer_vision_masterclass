# draw a rectangle on the displayed video 
# by inputting the top-left and bottom-right corners

import cv2


def draw_rectangle(event, x, y, flags, param):

    global pt1, pt2, topLeft_Clicked, botRight_Clicked

    if event == cv2.EVENT_LBUTTONDOWN:

        if topLeft_Clicked and botRight_Clicked:
            pt1 = (0, 0)
            pt2 = (0, 0)
            topLeft_Clicked = False
            botRight_Clicked = False
        if not topLeft_Clicked:
            pt1 = (x, y)
            topLeft_Clicked = True
        elif not botRight_Clicked:
            pt2 = (x, y)
            botRight_Clicked = True


pt1 = (0, 0)
pt2 = (0, 0)
topLeft_Clicked = False
botRight_Clicked = False

cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.setMouseCallback('Test', draw_rectangle)

while True:

    ret, frame = cap.read()

    if topLeft_Clicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0, 255, 0), thickness=-1)

    if topLeft_Clicked and botRight_Clicked:
        cv2.rectangle(frame, pt1, pt2, color=(0, 0, 255), thickness=3)

    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
