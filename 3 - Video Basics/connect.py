import cv2

capture = cv2.VideoCapture(0) # selects the default input

# the following values are returned as floats
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output the captured video onto disk
writer = cv2.VideoWriter('sample_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

while True:

    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Optional conversion to grayscale
    writer.write(frame)
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
writer.release()
cv2.destroyAllWindows()