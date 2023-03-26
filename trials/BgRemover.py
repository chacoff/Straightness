import cv2
from rembg import remove, new_session
import numpy as np


def log(*kwargs):
    return print(*kwargs)


def guardClause(**kwargs):
    if input.shape != output.shape:
        log(f'Shape issues between input and output image\ninput: {input.shape}\noutput: {output.shape}')
        return False
    return True


if __name__ == '__main__':

    log(f'cv2: {cv2.__version__}')

    modelName = new_session(model_name='u2netp')  # u2net, u2netp, silueta, u2net_human_seg, u2net_cloth_seg
    input_path = 'C:\\Coding\\801_Straightness\\images\\20211207_162959.jpg'
    output_path = 'C:\\Coding\\801_Straightness\\images\\20211207_0.png'

    input = cv2.imread(input_path)
    output = remove(input, session=modelName, post_process_mask=False, only_mask=False)[:, :, :3]  # disregarding alpha channel

    if guardClause():
        cv2.imshow('', cv2.resize(np.hstack((input, output)), None, fx=0.60, fy=0.60))
        cv2.waitKey(0)
        cv2.imwrite(output_path, output)
