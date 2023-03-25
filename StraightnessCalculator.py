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


def drawReturnMaxPoint(res, defects, canvas2draw, n_points=200):

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
            cv2.line(canvas2draw, start, end, [211, 84, 0], 1)
            cv2.circle(canvas2draw, far, 4, [211, 84, 0], -1)  # the max curvature
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem

            # arc_line = list(zip(*line(*vertexs[3], *vertexs[0])))  # faster but requires scikit-image
            arc_line = np.linspace(start, end, n_points, dtype=int)  # n_points samples on the line

    return far, round(angle, 2), arc_line


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

    return binaryfromcnt


def find4Vertex(canvas2draw, binaryfromcnt):
    # finding vertex with Shi Tomasi and good features to track
    corners = cv2.goodFeaturesToTrack(binaryfromcnt, 4, 0.01, 10, useHarrisDetector=False, k=0.04)  # set to find maximum 4 vertex
    corners = np.int0(corners)

    vertexs = []  # save all vertex and draw these
    for i in corners:
        x, y = i.ravel()
        vertexs.append((x, y))  # 0 BR - 1 TL - 2 TR - 3 BL
        cv2.circle(canvas2draw, (x, y), 3, [255, 50, 255], -1)

    return vertexs


def BorderDetectionCanny(border_channel):
    canny = cv2.Canny(border_channel, 30, 300)  # for border detection
    # [0] instead of using hierarchy for findContours()
    contours = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mainParameters():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
    ap.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
    ap.add_argument("-c", "--crop", type=int, default=0, help="whether to crop out largest rectangular region")
    ap.add_argument("-d", "--display", type=str2bool, help="display vertexes and bounding box")
    return vars(ap.parse_args())


def plot_image_grid(images, ncols=None, cmap='gray'):

    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1

    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]

    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=cmap)
        ax.axis("off")

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images2plot = []
    params = mainParameters()
    stiched_image = Stiching_images(params)
    images2plot.append(stiched_image)

    roi, blank_image, source = main(stiched_image, kern=3)  # args["output"] or images\\01.jpeg
    r, g, b = cv2.split(roi)[:3]  # splitting in RGB channels

    contours = BorderDetectionCanny(r)

    binarycnt = BinaryImgfromContour(blank_image, contours)
    images2plot.append(binarycnt)

    res, defects = FindHullDefects(contours)  # [cnt, defects]

    point, angle, arcLine = drawReturnMaxPoint(res, defects, blank_image, n_points=400)

    knot, dist = closest_node(point, arcLine)
    cv2.circle(blank_image, (knot[0], knot[1]), 3, [255, 50, 255], -1)
    print(f'[INFO] bending is {round(dist, 1)}px')

    cv2.drawContours(blank_image, contours, -1, (255, 255, 255), 1)  # [contours] to draw

    if params['display']:
        drawBoundingContour(blank_image, contours)
        find4Vertex(blank_image, binarycnt)
        drawApproxPolyDP(blank_image, res)

    images2plot.append(blank_image)
    plot_image_grid(images2plot, 1)
