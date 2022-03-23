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


class DiscreteTerrainClassifier(TerrainClassifier):
    def __init__(self, features=None, *args, **kwargs):
        self.knc = KNeighborsClassifier(*args, **kwargs)
        super().__init__(features)

    def feature_predict(self, feature_vec):
        return self.knc.predict_proba(feature_vec)[0]

    def fit(self, X, y):
        super().fit(X, y)
        self.knc.fit(X, y)


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