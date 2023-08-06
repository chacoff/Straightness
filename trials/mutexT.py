from time import sleep
from random import random
from threading import Thread
from threading import Lock
import onnxruntime
from u2netOnnx import run_inference
import cv2
import glob


def task(lock, identifier, onnx_sess, im):
    with lock:
        onnx_result = run_inference(onnx_sess, 320, im[identifier])
        mask = cv2.resize(onnx_result, dsize=(im[identifier].shape[1], im[identifier].shape[0]))
        name = f'>thread {identifier} got the lock'
        cv2.putText(mask, name, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.imwrite(f'tread_{identifier}.jpg', mask)
        print(f'>thread {identifier} got the lock')
        sleep(0.01)


def readImages(path):
    cv_img = []
    for img in glob.glob(path):
        n = cv2.imread(img)
        cv_img.append(n)
    return cv_img


if __name__ == '__main__':
    lock = Lock()
    onnx_session = onnxruntime.InferenceSession('C:\\Users\\gomezja\\.u2net\\u2net.onnx')

    im = readImages('..\\images\\beam\\*.bmp')

    for i in range(len(im)):
        Thread(target=task, args=(lock, i, onnx_session, im)).start()
