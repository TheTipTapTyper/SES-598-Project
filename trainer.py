from simanneal.anneal import Annealer
from terrain_cls import DiscreteTerrainClassifier, ContinuousTerrainClassifier
from numpy.random import default_rng
import numpy as np


class ClassifierTrainer(Annealer):
    def __init__(self, training_images, num_updates=1, grid_size=5, *args, **kwargs):
        """ Each training image is subdivided into a grid_size x grid_size equal
        regions. The features of each of these regions are extracted to create
        grid_size x grid_size x num_features numpy array.

        The state is a list of tuples with len(training_images elements). The 
        first element of each tuple is the 3D features array of one of the images.
        The second is a grid_size x grid_size 2D array where the value indicates 
        the class of the corresponding region. These class labels are the parameters 
        of the classifier and are what is updated by self.move.
        """
        self.classifier = self.load_classifier(*args, **kwargs)
        self.num_updates = num_updates
        self.grid_size = grid_size
        self.rng = default_rng()
        self.features_arrays = [
            self.subregion_feature_array(image) for image in training_images
        ]
        self.random_restart()

    def load_classifier(self):
        raise NotImplementedError()

    def random_restart(self):
        raise NotImplementedError()

    def new_param(old_param):
        raise NotImplementedError()

    def move(self):
        """ Changes the value of num_updates parmeters according to self.new_param
        based on their previous value. Returns the energy delta caused by the
        parameter update(s).
        """
        pass

    def energy_of_sample(self, sample_index):
        """ Calculates the energy contribution of the given image. This energy
        calculation is an implementation of the spatial continuity cost function
        found on page two of http://www.cim.mcgill.edu/~mrl/pubs/philg/crv2009.pdf.        
        """
        pass

    def energy(self):
        """ Sums the energy contributions of each training sample.
        """
        self.sample_energies = [self.energy_of_sample(idx) for idx in len(self.state)]
        return sum(self.sample_energies)

    def subregion_feature_array(self, image):
        rows, cols = image.shape[:2]
        assert rows % self.grid_size == 0 and cols % self.grid_size == 0, \
            'number of pixels in rows and columns of training images must be \
            divisible by grid_size: {}'.format(self.grid_size)
        gs = self.grid_size
        # subregion number of rows and columns
        sr_n_rows = rows // gs
        sr_n_cols = cols // gs
        features = np.zeros((gs, gs, self.classifier.num_features))
        for row_offset in range(gs):
            for col_offset in range(gs):
                subregion = image[
                    row_offset * sr_n_rows:(row_offset + 1) * sr_n_rows, 
                    col_offset * sr_n_cols:(col_offset + 1) * sr_n_cols
                ]
                features[row_offset, col_offset] = self.classifier.extract_features(subregion)
        return features


class DiscreteClassifierTrainer(ClassifierTrainer):
    def load_classifier(self, *args, **kwargs):
        return DiscreteTerrainClassifier(*args, **kwargs)

    def random_restart(self):
        self.state = [
            (fa, self.rng.integers(low=0, high=1, endpoint=True, size=(5,5))) for
            fa in self.features_arrays
        ]



if __name__ == '__main__':
    from PIL import Image    
    main_image = np.array(Image.open('simple_parking_lot.png'))[:,:,:3]
    training_image = main_image[1150:2250,1150:2250]
    Image.fromarray(training_image).show()
    dct = DiscreteClassifierTrainer([training_image])
    print('features: ', dct.state[0][0])
    print('classes: ', dct.state[0][1])