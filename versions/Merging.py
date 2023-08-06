import numpy as np
import cv2
import glob


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(img.shape[0] for img in img_list)

    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]), h_min),
                                 interpolation=interpolation)
                      for img in img_list]

    return cv2.hconcat(im_list_resize)


def main():
    return glob.glob('images/*.png')


if __name__ == "__main__":
    img_list = main()
    img_read = [cv2.imread(p) for p in img_list]
    
    img_h_resize = hconcat_resize(img_read)

    cv2.imshow('hconcat_resize.jpg', img_h_resize)
    cv2.waitKey()
