from simanneal.anneal import Annealer
from terrain_cls import DiscreteTerrainClassifier, ContinuousTerrainClassifier
from numpy.random import default_rng
import numpy as np
import datetime

class ClassifierTrainer(Annealer):
    def __init__(self, training_images, num_updates=1, grid_size=5, *args, **kwargs):
        """ Each training image is subdivided into a grid_size x grid_size equal
        regions. The features of each of these regions are extracted to create
        grid_size x grid_size x num_features numpy array.

        The state is a list of grid_size x grid_size 2D arrays, one for each training
        image. Each cell represents the class (or class probability) of the
        corresponding subregion in the image's feature array. These class labels are
        the parameters of the classifier and are what is updated by self.move.
        """
        self.classifier = self.load_classifier(*args, **kwargs)
        self.copy_strategy = 'method'
        self.num_updates = num_updates
        self.grid_size = grid_size
        self.rng = default_rng()
        self.features_arrays = [
            self.classifier.subregion_feature_array(
                image, self.grid_size
            ) for image in training_images
        ]
        self.random_restart()

    def load_classifier(self):
        raise NotImplementedError()

    def randomize_state(self):
        raise NotImplementedError()

    def new_param(self, old_param):
        raise NotImplementedError()
    
    def random_restart(self):
        self.randomize_state()
        self.update_classifier_params()

    def move(self):
        """  Updates the classifer with the new param(s) and
        returns the energy delta caused by the parameter update(s).
        """
        # effected_sample_idxs = self.update_state()
        self.update_state()
        self.update_classifier_params()
        # effected_old_energies = [
        #     self.sample_energies[i] for i in effected_sample_idxs
        # ]
        # for idx in effected_sample_idxs:
        #     energy = self.energy_of_sample(idx)
        #     self.sample_energies[idx] = energy
        # effected_new_energies = [
        #     self.sample_energies[i] for i in effected_sample_idxs
        # ]
        # new_e = sum(effected_new_energies)
        # old_e = sum(effected_old_energies)
        # delta_e = new_e - old_e
        # return delta_e

    def update_classifier_params(self):
        """ Fit the classifier with the current parameters in the state.
        """
        samples = []
        labels = []
        num_subregions = self.grid_size ** 2
        num_features = self.classifier.num_features
        for i in range(len(self.state)):
            samples.append(
                self.features_arrays[i].reshape(num_subregions, num_features)
            )
            labels.append(self.state[i].reshape(num_subregions, 1))
        X = np.vstack(samples)
        y = np.vstack(labels)
        self.classifier.fit(X, y)

    def update_state(self):
        """ Changes the value of num_updates parmeters according to self.new_param
        based on their previous value. Returns a list of the indeces of the effected
        samples.
        """
        num_samples = len(self.state)
        effected_sample_idxs = []
        for i in range(self.num_updates):
            sample_idx = self.rng.integers(num_samples)
            row = self.rng.integers(self.grid_size)
            col = self.rng.integers(self.grid_size)
            old_param = self.state[sample_idx][row, col]
            self.state[sample_idx][row, col] = self.new_param(old_param)
            effected_sample_idxs.append(sample_idx)
        return effected_sample_idxs

    def energy_of_sample(self, sample_index):
        """ Calculates the energy contribution of the given image. This energy
        calculation is an implementation of the spatial continuity cost function
        found on page two of http://www.cim.mcgill.edu/~mrl/pubs/philg/crv2009.pdf.        
        """
        # calculate probabilities of each subregion
        probs = np.zeros((self.grid_size, self.grid_size))
        num_features = self.classifier.num_features
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                probs[row, col] = self.classifier.feature_predict(
                    self.features_arrays[sample_index][row, col].reshape(1, num_features)
                )
        # reshape into vertical and horizontal continuity vectors
        horizontal_probs = probs.reshape(self.grid_size ** 2)
        vertical_probs = probs.T.reshape(self.grid_size ** 2)
        # calculate objective funciton (energy)
        numerator = 0
        num_subregions = self.grid_size ** 2
        for i in range(num_subregions):
            # ignore continuity gaps between row and column transitions
            if (i + 1) % self.grid_size != 0:
                # for class 1
                numerator += (horizontal_probs[i+1] - horizontal_probs[i]) ** 2
                numerator += (vertical_probs[i+1] - vertical_probs[i]) ** 2
                # for class 2 (1 - prob(class 1))
                numerator += ((1 - horizontal_probs[i+1]) - (1 - horizontal_probs[i])) ** 2
                numerator += ((1 - vertical_probs[i+1]) - (1 - vertical_probs[i])) ** 2
        denominator = np.var(probs) ** 2
        energy = numerator / denominator
        return energy

    def energy(self):
        """ Sums the energy contributions of each training sample.
        """
        self.sample_energies = [
            self.energy_of_sample(idx) for idx in range(len(self.state))
        ]
        return sum(self.sample_energies)

    def train(self, num_restarts=10):
        results = []
        for i in range(num_restarts):
            print('\n Run {}/{}'.format(i+1, num_restarts))
            best_state, best_energy = self.anneal()
            self.state = best_state
            self.update_classifier_params()
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".params"
            self.classifier.save(filename=fname)
            results.append((best_energy, fname))
        overall_best_energy, fname = results.sort(key=lambda x: x[0])[0]
        self.classifer.load(fname)
        print('Overall best energy: {}'.format(overall_best_energy))
        

class DiscreteClassifierTrainer(ClassifierTrainer):
    def load_classifier(self, *args, **kwargs):
        return DiscreteTerrainClassifier(*args, **kwargs)

    def randomize_state(self):
        self.state = [
            self.rng.integers(1, size=(self.grid_size,self.grid_size))
            for _ in self.features_arrays
        ]

    def new_param(self, old_param):
        """flip the bit"""
        return int(not old_param)

class ContinuousClassifierTrainer(ClassifierTrainer):
    def load_classifier(self, *args, **kwargs):
        return ContinuousTerrainClassifier(*args, **kwargs)

    def randomize_state(self):
        self.state = [
            self.rng.random(size=(5,5)) 
            for _ in self.features_arrays
        ]

    def new_param(self, old_param):
        """flip the bit"""
        return self.rng.random()


def train():
    from PIL import Image    
    main_image = np.array(Image.open('simple_parking_lot.png'))[:,:,:3]
    training_images = [
        main_image[1000:2000,1000:2000],
        main_image[0:1000,0:1000],
        main_image[3000:4000,1500:2500],
        main_image[4200:5200,4000:5000],
    ]
    #Image.fromarray(training_image).show()
    dct = DiscreteClassifierTrainer(training_images, num_updates=3)
    dct.train(num_restarts=3)
    mask = dct.classifier.segment_image(main_image)
    input('show image mask...')
    Image.fromarray(mask).show()


def train_continuous():
    from PIL import Image    
    main_image = np.array(Image.open('simple_parking_lot.png'))[:,:,:3]
    training_images = [
        main_image[1000:2000,1000:2000],
        main_image[0:1000,0:1000],
        main_image[3000:4000,1500:2500],
        main_image[4200:5200,4000:5000],
    ]
    #Image.fromarray(training_image).show()
    dct = ContinuousClassifierTrainer(training_images, num_updates=3)
    dct.train(num_restarts=3)
    mask = dct.classifier.segment_image(main_image)
    input('show image mask...')
    Image.fromarray(mask).show()

if __name__ == '__main__':
    #train()
    #determine_best_schedule()
    train_continuous()
