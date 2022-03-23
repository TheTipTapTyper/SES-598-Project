"""
Compare the run time of various methods of converting an RGB image in the form
of a numpy array to the HSV color space.

Results:
pillow_func: 1.4744 seconds
opencv_func: 0.3193 seconds
skimage_func: 7.0889 seconds

Conclusion: opencv is significantly faster

"""

import cv2
from skimage.color import rgb2hsv
from PIL import Image
import numpy as np

import timeit




def pillow_func(arr):
    return np.array(Image.fromarray(arr).convert('HSV'))

def opencv_func(arr):
    return cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)

def skimage_func(arr):
    return rgb2hsv(arr)


if __name__ == '__main__':
    img = Image.open('simple_parking_lot.png')
    arr = np.array(img)[:,:,:3]

    num = 5
    for func in [pillow_func, opencv_func, skimage_func]:
        avg_time = timeit.timeit(lambda: func(arr), number=num) / num
        print('{}: {:.4f} seconds'.format(func.__name__, avg_time))
