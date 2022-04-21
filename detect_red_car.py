"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 4/8/2022
Class: SES 598 Autonomous Exploration Systems
Project: Parking Lot Explorer
Description:
    This module provides a function which takes an rgb image and detects whether
    or not there is a red object (car) in the image and where (x, y).
"""

import cv2
import numpy as np

# bounds on saturation and value (HSV color space)
MIN_SAT = 100 # out of 255
MAX_SAT = 255
MIN_VAL = 60
MAX_VAL = 255

# lower and upper bounds for red hugh range
# red covers both high and low end of spectrum
LOW_RED_HUGH_LB = 0 # out of 255
LOW_RED_HUGH_UP = 20
HIGH_RED_HUGH_LB = 240
HIGH_RED_HUGH_UB = 255


def red_mask(img):
    """ Creates a numpy array with the same height and width of img but only
    one channel. This new array has a value of 1 where img has a red pixel and
    a 0 otherwise.
    """
    img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV_FULL)
    low_red_lb = np.array([LOW_RED_HUGH_LB, MIN_SAT, MIN_VAL])
    low_red_ub = np.array([LOW_RED_HUGH_UP, MAX_SAT, MAX_VAL])
    high_red_lb = np.array([HIGH_RED_HUGH_LB, MIN_SAT, MIN_VAL])
    high_red_ub = np.array([HIGH_RED_HUGH_UB, MAX_SAT, MAX_VAL])
    mask_low = cv2.inRange(img_hsv, low_red_lb, low_red_ub)
    mask_high = cv2.inRange(img_hsv, high_red_lb, high_red_ub)
    mask = mask_low + mask_high
    mask[mask > 0] = 1
    return mask


def apply_mask(img, mask):
    """ Removes the masked off portions of the image.
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img

def detect_red_obj(img, coverage_threshold=0.005):
    """ Detects whether or not a red object is in the image. If one is detected,
    the x and y coordinates are returned in the range [0,1] where (0,0) is the top
    left corner.

    presence of multiple red objects will lead to undefined behavior

    img: h x w x 3 array (rgb image)

    coverage_threshold: the percentage of pixels in the image that must be red in
        order for it to be considered a detection. If the coverage is below the
        threshold then None is returned.
    """
    mask = red_mask(img)
    coverage = mask.sum() / mask.size
    if coverage < coverage_threshold:
        return None
    y, x = np.array(np.where(mask == 1)).mean(axis=1)
    height, width = mask.shape
    x = x / width
    y = y / height
    return x, y

if __name__ == '__main__':
    img = cv2.imread('sim_car_close.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = red_mask(img_rgb)
    coords = detect_red_obj(img_rgb)
    masked_img = apply_mask(img, mask)

    if coords is not None:
        print('red object detected at ({:.3f}, {:.3f})'.format(
            coords[0], coords[1]
        ))
    else:
        print('No red object detected')

    cv2.imshow('image', img)
    cv2.imshow('masked', masked_img)
    while(True):
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break



