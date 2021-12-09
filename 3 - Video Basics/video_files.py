import cv2
import time

cap = cv2.VideoCapture('hand_move.mp4')

if not cap.isOpened():
    print('ERROR - FILE NOT FOUND, OR WRONG CODEC USED')

# ret gives the status of the recording process as a boolean value
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        time.sleep(1/20)  # sets the display frame rate - not useful for processing
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
