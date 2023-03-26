import numpy as np
import cv2

video = cv2.VideoCapture('images\\video.mp4')

kernel = None
backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = video.read()
    if not ret:
        break

    fgmask = backgroundObject.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) >= 600:  # noise threshold
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(frameCopy, 'Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)

    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
    stacked = np.hstack((foregroundPart, frameCopy))
    cv2.imshow('Detected', cv2.resize(stacked, None, fx=0.25, fy=0.25))
    k = cv2.waitKey(1) & 0xff

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()