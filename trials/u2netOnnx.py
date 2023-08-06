import os
import copy
import time
import cv2
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size, input_size))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1).astype('float32')
    x = x.reshape(-1, 3, input_size, input_size)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result


def unsharp_mask(image, kernel_size=(5, 5), sigma=3.0, amount=2.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == '__main__':

    # Parameters
    postpro = False
    pad = 100
    input_size = 320  # expected by u2
    image_path = '..\\images\\20211207_162959.jpg'
    name = '..\\images\\d'+image_path.split('\\')[-1]
    model_path = 'C:\\Users\\jaime\\.u2net\\u2net.onnx'
    onnx_session = onnxruntime.InferenceSession(model_path)

    # Image Pre-processing
    tic = time.perf_counter()

    image_raw = cv2.imread(image_path)
    image_pad = cv2.copyMakeBorder(image_raw, 0, 0, pad, pad, borderType=cv2.BORDER_CONSTANT)
    image_pad = unsharp_mask(image_pad)
    image = change_brightness(image_pad, value=-10)

    # Image to the network
    onnx_result = run_inference(onnx_session, input_size, image)

    toc = time.perf_counter()
    elapsed_time_text = f'elapsed: {toc - tic:0.2f}s'

    mask = cv2.resize(onnx_result, dsize=(image.shape[1], image.shape[0]))

    """
        Post Process the mask for a smooth boundary by applying Morphological Operations
        Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    """
    if postpro:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (3, 3))
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
        mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # convert again to binary

    res_image = cv2.bitwise_and(image, image, mask=mask)

    # Unpadding resulting image
    res_image = res_image[:, pad:res_image.shape[1]-pad]  # [y:y+h, x:x+w]
    res_image = change_brightness(res_image, value=10)

    cv2.putText(image_raw, elapsed_time_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('result with U2net', cv2.resize(np.vstack((image_raw, res_image)), None, fx=0.25, fy=0.25))
    # cv2.imwrite(name, res_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()