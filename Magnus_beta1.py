import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage.draw import line
from scipy.spatial import distance
import argparse
import imutils
from Stiching import Stiching_images


def FindHullDefects(segment):
    # _, contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = segment

    # find largest area contour
    max_area = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            cnt = contours[i]
            max_area = area

    cnt = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    return cnt, defects


def drawApproxPolyDP(canvas2draw, res):
    for p in res:
        x, y = p.ravel()
        cv2.circle(canvas2draw, (x, y), 4, [0, 255, 255], -1)


def drawReturnMaxPoint(res, defects, canvas2draw):

    if type(defects) != type(None):  # avoid crashing
        # cnt = 0
        for i in range(defects.shape[0]):  # calculate the angle
            s, e, f, d = defects[i][0]
            start = tuple(res[s][0])
            end = tuple(res[e][0])
            far = tuple(res[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            cv2.circle(canvas2draw, far, 4, [211, 84, 0], -1)  # the max curvature
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem

            # if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
            #    cnt += 1
            #    cv2.circle(blank_image, far, 8, [211, 84, 0], -1)

    return far, round(angle, 2)


def drawBoundingContour(canvas2draw, contours):
    print(f'[INFO] Contours found: {len(contours)}')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(canvas2draw, (x, y), (x+w, y+h), (0, 255, 0), 1)  # bounding box


def BinaryImgfromContour(canvas2draw, contour):
    filled_contour = cv2.drawContours(canvas2draw.copy(),
                                      [max(contour, key=cv2.contourArea)],
                                      -1,
                                      (255, 255, 255),
                                      thickness=cv2.FILLED)

    binaryfromcnt = cv2.split(filled_contour)[:3][0]  # splitting in RGB channels
    cv2.imwrite('images\\debugs\\filled.png', binaryfromcnt)

    return binaryfromcnt


def find4Vertex(canvas2draw, binaryfromcnt, display_info=False, n_points=180):
    # finding vertex with Shi Tomasi and good features to track

    corners = cv2.goodFeaturesToTrack(binaryfromcnt, 4, 0.01, 10, useHarrisDetector=False, k=0.04)  # set to find maximum 4 vertex
    corners = np.int0(corners)

    vertexs = []  # save all vertex and draw these
    for i in corners:
        x, y = i.ravel()
        vertexs.append((x, y))  # 0 BR - 1 TL - 2 TR - 3 BL
        if display_info:
            cv2.circle(canvas2draw, (x, y), 3, [255, 50, 255], -1)

    # arc_line = list(zip(*line(*vertexs[3], *vertexs[0])))  # faster but requires scikit-image
    arc_line = np.linspace(vertexs[2], vertexs[1], n_points, dtype=int)  # 100 samples on the line

    if display_info:
        for pl in arc_line:
            cv2.circle(canvas2draw, (pl[0], pl[1]), 1, [0, 255, 255], -1)

    '''''''''        
        print(f'{len(vertexs)} vertex\n '
              f'BottomRight: {vertexs[0]}\n '
              f'TopLeft: {vertexs[1]}\n '
              f'TopRight: {vertexs[2]}\n '
              f'BottomLeft: {vertexs[3]}')
    cv2.line(canvas2draw, vertexs[2], vertexs[1], (0, 0, 255), thickness=1, lineType=8)
    '''''''''

    return vertexs, arc_line


def BorderDetectionCanny(border_channel):
    canny = cv2.Canny(border_channel, 30, 300)  # for border detection
    # [0] instead of using hierarchy for findContours()
    contours = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def UniqueBottomProfile(contours, corns, canvas2draw):  # trying to get rid of this unique one time method
    contorno = np.vstack(contours).squeeze()  # stack them together followed by squeeze to remove redundant axis.
    bottom_profile = []
    for i in range(contorno.shape[0]):
        if contorno[i][1] >= corns[3][1]:
            bottom_profile.append(list(contorno[i]))
        if contorno[i][0] >= corns[0][0]:
            break
    bottom_profile = np.array(bottom_profile, np.int32)
    bottom_profile = bottom_profile.reshape((-1, 1, 2))
    cv2.drawContours(canvas2draw, bottom_profile, -1, (0, 255, 255), 2)  # [bottom_profile] to draw


def mask2HSV(roi_image):
    # HSV
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([8, 30, 130])
    upper_red = np.array([20, 90, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)  # Threshold the HSV image to get only blue colors
    res = cv2.bitwise_and(roi_image, roi_image, mask=mask)  # Bitwise-AND mask and original image
    return res


def closest_node(node, nodes):
    closest = distance.cdist([node], nodes)
    index = closest.argmin()
    euclidean = closest[0]
    return nodes[index], euclidean[index]


def main(image_dir, kern=3):
    # source = cv2.imread(image_dir)
    source = image_dir

    height, width, channels = source.shape
    blank_image = np.zeros((height, width, channels), np.uint8)  # canvas2draw

    median = cv2.medianBlur(source.copy(), kern)  # Filters to denoise the image
    denoise = cv2.GaussianBlur(median, (kern, kern), 0)
    roi = denoise  # [360:590, 6:1250]  # y:y+h, x:x+w  # ROI if needed

    return roi, blank_image, source


def mainParameters():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
    ap.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
    ap.add_argument("-c", "--crop", type=int, default=0, help="whether to crop out largest rectangular region")
    return vars(ap.parse_args())


if __name__ == '__main__':
    params = mainParameters()
    stiched_image = Stiching_images(params)

    roi, blank_image, source = main(stiched_image, kern=3)  # args["output"] or images\\01.jpeg
    r, g, b = cv2.split(roi)[:3]  # splitting in RGB channels

    contours = BorderDetectionCanny(r)
    drawBoundingContour(blank_image, contours)

    binarycnt = BinaryImgfromContour(blank_image, contours)

    verxs, arcLine = find4Vertex(blank_image, binarycnt, display_info=True, n_points=400)

    res, defects = FindHullDefects(contours)  # [cnt, defects]
    drawApproxPolyDP(blank_image, res)

    point, angle = drawReturnMaxPoint(res, defects, blank_image)

    knot, dist = closest_node(point, arcLine)
    cv2.circle(blank_image, (knot[0], knot[1]), 3, [255, 50, 255], -1)
    print(f'[INFO] bending is {round(dist, 1)}px')

    cv2.drawContours(blank_image, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Result', blank_image)
    cv2.waitKey()
