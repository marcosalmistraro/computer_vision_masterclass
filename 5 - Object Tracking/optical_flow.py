import cv2
import numpy as np

# sparse points with the Lucas-Kanade method
corner_track_params = dict(maxCorners=10,
                           qualityLevel=0.3,
                           minDistance=7,
                           mask=None,
                           blockSize=7)
lk_params = dict(winSize=(200, 200),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# pick the points to be tracked
prevPts = cv2.goodFeaturesToTrack(prev_gray, **corner_track_params)
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
    # only select points where the status vector is 1, meaning motion is detected
    good_new = nextPts[status == 1]
    good_prev = prevPts[status == 1]
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3)
        frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('Tracking', img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
