# example of face and eyes detection in cv2 using haar cascades

import cv2
import matplotlib.pyplot as plt


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


denis = cv2.imread('denis_mukwege.jpg', 0)
nadia = cv2.imread('nadia_murad.jpg', 0)
solvay = cv2.imread('solvay_conference.jpg', 0)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

nadia_face = detect_face(nadia)
denis_face = detect_face(denis)
nadia_eyes = detect_eyes(nadia)
denis_eyes = detect_eyes(denis)
solvay_faces = adjusted_detect_face(solvay)
solvay_eyes = detect_eyes(solvay)

plt.imshow(nadia_face, cmap='gray')
plt.show()
plt.imshow(nadia_eyes, cmap='gray')
plt.show()
plt.imshow(denis_face, cmap='gray')
plt.show()
plt.imshow(denis_eyes, cmap='gray') # this doesn't work well
plt.show()
plt.imshow(solvay_faces, cmap='gray')
plt.show()
plt.imshow(solvay_eyes, cmap='gray') # this also fails to detect eyes
plt.show()
