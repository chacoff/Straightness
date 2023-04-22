import cv2
from rembg import remove, new_session
import numpy as np
import time
from u2netOnnx import unsharp_mask


def log(*kwargs):
    return print(*kwargs)


def guardClause(**kwargs):
    if input.shape != output.shape:
        log(f'Shape issues between input and output image\ninput: {input.shape}\noutput: {output.shape}')
        return False
    return True


if __name__ == '__main__':

    log(f'cv2: {cv2.__version__}')

    # On initialization
    modelName = new_session(model_name='u2net')  # u2net, u2netp, silueta, u2net_human_seg, u2net_cloth_seg
    input_path = '..\\images\\20211207_162959.jpg'
    output_path = '..\\images\\20211207_1.png'

    tic = time.perf_counter()
    input_raw = cv2.imread(input_path)
    input = unsharp_mask(input_raw)
    output = remove(input, session=modelName, post_process_mask=False, only_mask=False)[:, :, :3]  # disregarding alpha channel
    toc = time.perf_counter()

    if guardClause():
        cv2.imwrite(output_path, output)
        log(f'elapsed: {toc - tic:0.2f}s')
        cv2.imshow('', cv2.resize(np.vstack((input, output)), None, fx=0.35, fy=0.28))
        cv2.waitKey(0)

