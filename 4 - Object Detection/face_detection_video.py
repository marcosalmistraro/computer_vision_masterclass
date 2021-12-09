import cv2


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return face_img


def adjusted_detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return face_img


def detect_eyes(img):
    eye_img = img.copy()
    eye_rects = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in eye_rects:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return eye_img


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read(0)
    frame = adjusted_detect_face(frame)
    frame = detect_eyes(frame)
    cv2.imshow('Video Face Detect', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
