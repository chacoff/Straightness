import cv2
from StraightnessCalculator import mask2HSV
from StraightnessCalculator import BorderDetectionCanny, BinaryImgfromContour
import numpy as np


image = cv2.imread('images/4.tif')
image_resized = cv2.resize(image, [640, 640], cv2.INTER_CUBIC)
r, g, b = cv2.split(image_resized)[:3]
height, width, channels = image_resized.shape
blank_image = np.zeros((height, width, channels), np.uint8)  # canvas2draw

can = BorderDetectionCanny(g)
to_show = BinaryImgfromContour(blank_image, can)

cv2.imshow('hsv', to_show)
cv2.waitKey()




