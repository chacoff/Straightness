import cv2
from StraightnessCalculator import mask2HSV
from StraightnessCalculator import BorderDetectionCanny, BinaryImgfromContour
import numpy as np


def bs():
    # cap = cv2.VideoCapture('./img/vtest.avi')

    # fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    fgbg = cv2.createBackgroundSubtractorKNN(history=3, detectShadows=True)

    # while cap.isOpened():
    for i in range(0, 4):
        # ret, frame = cap.read()
        ret, frame = True, cv2.imread('images\\20211207_162959.jpg')
        r, g, b = cv2.split(frame)
        cv2.putText(r, f'frame {i}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

        if not ret:
            break

        fgmask = fgbg.apply(b)
        stacked = np.hstack((r, fgmask))

        cv2.imshow(f'frames', cv2.resize(stacked, None, fx=0.60, fy=0.60))
        cv2.waitKey()

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    log = lambda *args: print(*args)
    bs()


'''
image = cv2.imread('images/4.tif')
image_resized = cv2.resize(image, [640, 640], cv2.INTER_CUBIC)
r, g, b = cv2.split(image_resized)[:3]
height, width, channels = image_resized.shape
blank_image = np.zeros((height, width, channels), np.uint8)  # canvas2draw

can = BorderDetectionCanny(g)
to_show = BinaryImgfromContour(blank_image, can)

cv2.imshow('hsv', to_show)
cv2.waitKey()
'''




