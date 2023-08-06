import cv2
import imageio
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt


source = cv2.imread('images/01.jpeg')
source_roi = source  #[360:590, 6:1250]  # y:y+h, x:x+w if ROI needed
height, width, channels = source_roi.shape
blank_image = np.zeros((height, width, channels), np.uint8)

# Filters to denoise the image
median = cv2.medianBlur(source_roi, 5)
denoise = cv2.GaussianBlur(median, (5, 5), 0)
r, g, b = cv2.split(denoise)[:3]  # splitting in RGB

# Border detection
canny = cv2.Canny(r, 30, 300)  # for border detection coming from _denoise_
contours = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # , hierarchy instead of [0]
contorno = np.vstack(contours).squeeze()  # stack them together followed by squeeze to remove redundant axis.

''''''''' not use if background is contrasted
# HSV
hsv = cv2.cvtColor(denoise, cv2.COLOR_BGR2HSV)
lower_red = np.array([8, 30, 130])
upper_red = np.array([20, 90, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)  # Threshold the HSV image to get only blue colors

# Bitwise-AND mask and original image
res = cv2.bitwise_and(denoise, denoise, mask=mask)
'''''''''

# finding vertex with Shi Tomasi and good features to track
corners = cv2.goodFeaturesToTrack(r, 4, 0.01, 10)  # set to find maximum 4 vertex in the Red channel of _denoise_
corners = np.int0(corners)

corns = []  # save all vertex and draw these
k = 0
for i in corners:
    x, y = i.ravel()
    cv2.circle(source_roi, (x, y), 2, [0, 0, 255], -1)
    corns.append((x, y))
    k = k + 1  # 0 BR - 1 TL - 2 TR - 3 BL

print(f'{k} vertex\n BottomRight: {corns[0]}\n TopLeft: {corns[1]}\n TopRight: {corns[2]}\n BottomLeft: {corns[3]}')
projection = (corns[0][0], corns[3][1])
print(f'Vertex to join with a red line {corns[3]} {corns[0]}, projection in a white line: {projection}')
# real_width = np.linalg.norm(corns[1] - corns[3])
# real_length = np.linalg.norm(cornsp[3] - corns[0])

# Drawings
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(blank_image, (x, y), (x+w, y+h), (0, 255, 0), 1)  # bounding box

cv2.drawContours(source_roi, contours, -1, (255, 255, 255), 1)
# cv2.line(blank_image, corns[3], corns[0], (0, 0, 255), thickness=1, lineType=8)  # projection between extremes
# cv2.line(blank_image, corns[3], projection, (0, 255, 255), thickness=1, lineType=8)  # projection from Bottom Left

bottom_profile = []
for i in range(contorno.shape[0]):
    if contorno[i][1] >= corns[3][1]:
        bottom_profile.append(list(contorno[i]))
    if contorno[i][0] >= corns[0][0]:
        break

points2fit = bottom_profile
bottom_profile = np.array(bottom_profile, np.int32)
bottom_profile = bottom_profile.reshape((-1, 1, 2))
cv2.drawContours(source_roi, [bottom_profile], -1, (0, 255, 255), 1)  # [bottom_profile] to draw

cv2.imshow('', source_roi)
cv2.waitKey()

cv2.drawContours(blank_image, bottom_profile, -1, (0, 255, 255), 1)  # [bottom_profile] to draw
cv2.imshow('', blank_image)

cv2.waitKey()

# polyfit
y_shift = corns[0][1] - corns[3][1]
x_axis = [0]  # we force 0
y_axis = [y_shift]  # we force 0
for p in range(len(points2fit)):
    x_axis.append(points2fit[p][0])
    y_axis.append(-points2fit[p][1]+corns[0][1])  # - because of the 0,0 from the image in compare to the 0,0 cartesian

z = poly.polyfit(x_axis, y_axis, 5)  # A + Bx + Cx^2 + Dx^3
ffit = poly.polyval(x_axis, z)
plt.plot(x_axis, ffit)
plt.show()