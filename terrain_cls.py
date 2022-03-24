"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/22/2022
Class: SES 598 Autonomous Exploration Systems
Project: Parking Lot Explorer

This file implements the terrain classifier/segmenter described in 
http://www.cim.mcgill.edu/~mrl/pubs/philg/crv2009.pdf. Some parts of the algorithm
are unclear and are imporvised as necessary.
"""
import pickle
from tkinter.filedialog import test

import numpy as np
from PIL import Image
import cv2
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from abc import ABC, abstractmethod


class TerrainClassifier(ABC):
    """ Classifies an image as being one of two types of terrain.
    """
    def __init__(self, features=None, *args, **kwargs):
        """ features is a string that indicates what information should
        be extracted from input images. Valid values:
        "rbg": the averages of the r, g and b channels
        "hsv": the average of the h, s, and v channels
        default: rgb
        """
        if features is None:
            features = 'rgb'
        if features == 'rgb':
            self.extract_features = extract_avg_rgb
            self.num_features = 3
        elif features == 'hsv':
            self.extract_features = extract_avg_hsv
            self.num_features = 3
        else:
            raise ValueError('feature_extractor {} not recognized'.format(features))
        self.params = None

    def image_predict(self, image):
        """ Predicts the class of the image. The output is a numpy array
        """
        return self.feature_predict(self.extract_features(image))

    @abstractmethod
    def feature_predict(self, features):
        pass

    def fit(self, X, y):
        """ Set the training set to be X and y. X is an n x num_features numpy
        array, and y is an n x num_classes numpy array where n is the number of
        samples. The rows of X are points in feature space.
        """
        self.params = [X, y]

    def save(self, filename='params.pickle'):
        """ Saves parameters to file.
        """
        assert self.params is not None, 'Params have not yet been initialized.'
        with open(filename, 'wb') as file:
            pickle.dump(self.params, file)

    def load(self, filename='params.pickle'):
        """ Loads parameters from file.
        """
        with open(filename, 'rb') as file:
            X, y = pickle.load(file)
            self.fit(X, y)

    def segment_image(self, image, grid_size=150, binary=False, flip_scale=False,
                      fullsize=False):
        """ Creates an segmentation mask (grey scale image). The values of 
        the mask are determined by the predicted terrain class of the 
        corresponding block of pixels.
        grid_size (int): number of grid squares down and accross the image
            is divided into. image width and height must be divisible by grid_size
        binary (bool): If true, blocks will be classified as class 0 if the
            predicted probability is < .5 and class 1 otherwise. This means
            that the mask's values will either be 0 or 255.
            If not, the color of the mask's values will be proportional to the
            predicted probability
        flip_scale (bool): whether or not to switch which class is dark and which
            is light.
        fullsize (bool): if true, mask is resized to the dimensions of the input
            image, else it will be grid_size x grid_size
        """
        features = self.subregion_feature_array(image, grid_size)
        mask = np.zeros((grid_size, grid_size))
        for row in range(grid_size):
            for col in range(grid_size):
                mask[row, col] = self.feature_predict(
                    features[row, col].reshape(1, self.num_features)
                )
                if flip_scale:
                    mask[row, col] = 1 - mask[row, col]
        if binary:
            mask[mask < .5] = 0
            mask[mask >= .5] = 1
        mask = np.interp(mask, [0,1], [0, 255])
        mask = np.array(mask, dtype=np.uint8)
        height, width = image.shape[:2]
        if fullsize:
            mask = cv2.resize(mask, (height, width))
        return mask

    def subregion_feature_array(self, image, grid_size):
        """ Divides an image into grid_size x grid_size subregions and extracts
        the features of each region. Return value is a grid_size x grid_size x
        num_features array.
        """
        rows, cols = image.shape[:2]
        assert rows % grid_size == 0 and cols % grid_size == 0, \
            'number of pixels in rows and columns of training images must be \
            divisible by grid_size: {}'.format(grid_size)
        gs = grid_size
        # subregion number of rows and columns
        sr_n_rows = rows // gs
        sr_n_cols = cols // gs
        features = np.zeros((gs, gs, self.num_features))
        for row_offset in range(gs):
            for col_offset in range(gs):
                subregion = image[
                    row_offset * sr_n_rows:(row_offset + 1) * sr_n_rows, 
                    col_offset * sr_n_cols:(col_offset + 1) * sr_n_cols
                ]
                features[row_offset, col_offset] = self.extract_features(subregion)
        return features


class DiscreteTerrainClassifier(TerrainClassifier):
    def __init__(self, features=None, *args, **kwargs):
        self.knc = KNeighborsClassifier(*args, **kwargs)
        super().__init__(features)

    def feature_predict(self, feature_vec):
        return self.knc.predict_proba(feature_vec)[0,0]

    def fit(self, X, y):
        super().fit(X, y)
        self.knc.fit(X, y.reshape(len(y)))


class ContinuousTerrainClassifier(TerrainClassifier):
    def __init__(self, features=None, *args, **kwargs):
        self.knr = KNeighborsRegressor(*args, **kwargs)
        super().__init__(features)

    def feature_predict(self, feature_vec):
        return self.knr.predict(feature_vec)

    def fit(self, X, y):
        super().fit(X, y)
        self.knr.fit(X, y)



def extract_avg_rgb(image):
    """ input: hxwx3 numpy array representing an RGB image
        returns: 1x3 numpy array where the elements are the averages of the 
        r g and b channels respectively
    """
    return image.mean(axis=(0,1))

def extract_avg_hsv(image):
    """ input: hxwx3 numpy array representing an RGB image
        returns: 1x3 numpy array where the elements are the averages of the 
        h s and v channels respectively
    """
    hsv_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    return hsv_image.mean(axis=(0,1))

def test_segment():
    filename = '/home/josh/code/github/SES-598-Project/2022-03-23T21h31m35s_energy_1384.6755241184524.params'
    from PIL import Image    
    main_image = np.array(Image.open('simple_parking_lot.png'))[:,:,:3]
    classifier = DiscreteTerrainClassifier()
    classifier.load(filename)
    mask = classifier.segment_image(main_image, grid_size=130, binary=True)
    Image.fromarray(mask).show()

if __name__ == '__main__':
    test_segment()