from simanneal.anneal import Annealer
from terrain_cls import TerrainClassifier
from numpy.random import default_rng
import numpy as np
import cv2



class ClassifierTrainer(Annealer):
    def __init__(self, steps=2000, discrete=False, num_updates=1, grid_size=5, 
                 *args, **kwargs):
        """ Each training image is subdivided into a grid_size x grid_size equal
        regions. The features of each of these regions are extracted to create
        grid_size x grid_size x num_features numpy array.

        The state is a list of grid_size x grid_size 2D arrays, one for each training
        image. Each cell represents the class (or class probability) of the
        corresponding subregion in the image's feature array. These class labels are
        the parameters of the classifier and are what is updated by self.move.

        if discrete, parameters (class probabilities) are treated as binary and
        an update to one will simply flip it. Thus the search problem becomes
        combinatorial. Else, class probabilities are updated by setting them to a
        random value between [0,1] and the search problem is continuous.
        """
        self.classifier = TerrainClassifier(*args, **kwargs)
        self.num_updates = num_updates
        self.grid_size = grid_size
        self.rng = default_rng()
        self.discrete = discrete
        
        #annealing params
        self.steps = steps
        self.copy_strategy = 'deepcopy'
        self.Tmax = 7000 # point at which acceptance drops below 98%

    def load_images(self, training_images):
        self.features_arrays = [
            self.classifier.subregion_feature_array(
                image, self.grid_size
            ) for image in training_images
        ]
        self.random_restart()

    def randomize_state(self):
        if self.discrete:
            self.state = [
                self.rng.integers(low=0, high=1, endpoint=True,
                size=(self.grid_size,self.grid_size))
                for _ in self.features_arrays
            ]
        else:
            self.state = [
                self.rng.random(size=(self.grid_size,self.grid_size))
                for _ in self.features_arrays
            ]

    def new_param(self, old_param):
        return int(not old_param) if self.discrete else self.rng.random()
    
    def random_restart(self):
        self.randomize_state()
        self.update_classifier_params()
        self.best_energy = None
        self.best_state = None

    def move(self):
        """  Updates the classifer with the new param(s) and
        returns the energy delta caused by the parameter update(s).
        """
        self.update_state()
        self.update_classifier_params()

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

    def sample_probs(self, sample_index):
        """ Compute the grid_size x grid_size array of probabilities of each
        subregion for the specified sample.
        """
        probs = np.zeros((self.grid_size, self.grid_size))
        num_features = self.classifier.num_features
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                probs[row, col] = self.classifier.feature_predict(
                    self.features_arrays[sample_index][row, col].reshape(1, num_features)
                )
        return probs

    def sample_continuity_estimate(self, probs):
        """ Calculates a value that estimates the level of spatial continuity
        (terrain class probabilities) of the sample.

        Continuity is measured as the difference is predicted class probabilities
        is spatially adjacent subregions of pixels.
        """
        # reshape into vertical and horizontal continuity vectors
        horizontal_probs = probs.reshape(self.grid_size ** 2)
        vertical_probs = probs.T.reshape(self.grid_size ** 2)
        # calculate objective funciton (energy)
        continuity = 0
        num_subregions = self.grid_size ** 2
        for i in range(num_subregions):
            # ignore continuity gaps between row and column transitions
            if (i + 1) % self.grid_size != 0:
                # for class 1
                continuity += (horizontal_probs[i+1] - horizontal_probs[i]) ** 2
                continuity += (vertical_probs[i+1] - vertical_probs[i]) ** 2
                # for class 2 (1 - prob(class 1))
                continuity += ((1 - horizontal_probs[i+1]) - (1 - horizontal_probs[i])) ** 2
                continuity += ((1 - vertical_probs[i+1]) - (1 - vertical_probs[i])) ** 2
        return continuity

    def energy(self):
        """ Calculates the energy of the system (cost associated with the current
        parameters). Reference http://www.cim.mcgill.edu/~mrl/pubs/philg/crv2009.pdf.
        """
        all_probs = [self.sample_probs(idx) for idx in range(len(self.state))]
        numerator = sum(self.sample_continuity_estimate(probs) for probs in all_probs)
        denominator = np.var(all_probs) ** 2
        total_energy = numerator / denominator
        # average energy 
        num_counted_subregion_boundaries = len(self.state) * (self.grid_size - 1) ** 2
        avg_energy = total_energy / num_counted_subregion_boundaries
        return avg_energy

    def train(self, num_restarts=10):
        results = []
        for i in range(num_restarts):
            self.random_restart()
            self.anneal()
            self.state = self.best_state
            self.update_classifier_params()
            print('\nrun {}/{} energy: {:.3f}'.format(i+1, num_restarts, self.best_energy))
            mode = 'dct' if self.discrete else 'con'
            path = 'params/en_{:.2f}_md_{}_gs_{}_nu_{}_fe_{}.params'.format(
                self.best_energy, mode, self.grid_size, self.num_updates, 
                self.classifier.features
            )
            self.classifier.save(filename=path)
            results.append((self.best_energy, path))
        results.sort(key=lambda x: x[0])
        overall_best_energy, fname = results[0]
        self.classifier.load(fname)
        print('Overall best energy: {}'.format(overall_best_energy))

def train():
    discrete=False
    grid_size=5
    steps=30000
    num_updates=1
    features='both'

    print('Discrete: {} grid_size: {} steps: {} num_updates: {} features: {}'.format(
        discrete, grid_size, steps, num_updates, features
    ))

    num_restarts = 10

    from PIL import Image    
    main_image = np.array(Image.open('simple_parking_lot.png'))[:,:,:3]
    training_images = [
        main_image[1000:2000,1000:2000],
        main_image[0:1000,0:1000],
        main_image[3000:4000,1500:2500],
        main_image[4200:5200,4000:5000],
    ]

    #Image.fromarray(training_image).show()
    dct = ClassifierTrainer(num_updates=num_updates, discrete=discrete, 
        grid_size=grid_size, steps=steps, features=features)
    dct.load_images(training_images)
    dct.train(num_restarts=num_restarts)
    mask = dct.classifier.segment_image(main_image, grid_size=130)
    input('show image mask...')
    Image.fromarray(mask).show()


if __name__ == '__main__':
    train()
