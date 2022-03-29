"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/26/2022
Class: SES 598 Autonomous Exploration Systems
Project: Parking Lot Explorer

This file implements a drone controller class which uses a trained terrain
segmenter in order to stay over the top of a parking lot in order to locate
a car. The controller is a finite state machine (FSM) which moves in a straight
line while over the parking lot and begins to spiral when it finds itself over
the desert until it is once again over the lot. Once it does, it will counter
turn by a random degree for the sake of more even exploration/lot coverage.
"""

from terrain_cls import TerrainClassifier
from PIL import Image
import numpy as np
from numpy.random import default_rng
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import tqdm

# FSM states
GO_STRAIGHT = 'GoStraight'
FIND_LOT = 'FindLot'
COUNTER_TURN = 'CounterTurn'

MIN_COUNTER_TURN_STEPS = 1
MAX_COUNTER_TURN_STEPS = 9

# conversion from meters to pixels
# 25px per meter. determined using known size of parking space
PX_PER_M = 25 

# distance traveled per iteration/step
M_PER_STEP = 1

# pixels traveled per iteration/step
DELTA_V_PX = M_PER_STEP * PX_PER_M

# max turning rate (tightest curve)
MAX_DELTA_THETA = 15 # degrees
MAX_COUNTER_TURN_DELTA_THETA = 5
# how quickly curve radius increases
DELTA_THETA_DECAY_RATE = .01 # percent decay per iteration

# starting position is the center of the 6500 pixel square image
START_X = START_Y = 3750
# starting direction is due east (90 is north, 180 is west, etc.)
START_THETA = 0

TERRAIN_FILE = 'complex_parking_lot.png'

#the number of prior predictions to maintain to average over
NUM_PRIOR_PREDICTIONS = 1

FOV = 5

TURNS_PER_DIRECTION = 10

NUM_ANIMATION_FRAMES = 200



class DroneController:
    def __init__(self, params_path, features, terrain_fn=TERRAIN_FILE, fov=FOV, lot_class=1):
        """ params_path (str): path to pickled set of classifier parameters
            features (str): a string denoting which feature extractor the 
                classifier should use. Ensure this corresponds to the feature
                extractor used when training this particular set of parameters.
            fov (int): width and height in meters of the field of view of the 
                drone's downward facing camera. 
            lot_class (int, 0 or 1): The training process of the classifier does 
                not ensure that the positive class (parking lot) is always 1,
                so, it is necessary to specifiy how this set of parameters handles
                the classification.
        """
        self.state = GO_STRAIGHT
        # calculate the size of the field of view in pixels
        self.fov = int(fov * PX_PER_M)
        # offsets needed to extract fov from terrain. necessary to support odd fov's
        self.low_offset = math.floor(self.fov / 2)
        self.high_offset = math.ceil(self.fov / 2)
        self.flip_classes = not bool(lot_class)
        self.classifier = TerrainClassifier(features=features)
        self.classifier.load(params_path)
        self.x = START_X
        self.y = START_Y
        self.theta = START_THETA
        self.delta_theta = 0
        # list maintaining history of poses (position and heading)
        self.path = [(self.x, self.y, self.theta)]
        self.load_terrain(terrain_fn)
        # initalize random number generator
        self.rng = default_rng()
        self.prior_predictions = []
        self.turns_since_dir_change = 0
        self.turn_direction = 1

    def load_terrain(self, filename):
        img = Image.open(filename)
        # remove alpha channel
        self.terrain = np.array(img)[:,:,:3]
        self.rows, self.cols = self.terrain.shape[:2]

    def move(self):
        """ update heading based on delta_theta and move at constant velocity.
        """
        self.theta += self.turn_direction * self.delta_theta
        # make sure theta stays in range [0,360)
        self.theta = self.theta % 360
        radians = self.theta * math.pi / 180
        self.x += DELTA_V_PX * math.cos(radians)
        self.y += DELTA_V_PX * math.sin(radians)

    @property
    def camera_view(self):
        """ current view of the downward facing drone camera. An fov x fov 2d 
        numpy array subimage taken from the terrain centered at the current
        position (self.x, self.y)
        """
        # need to offset y coordinate because x-y origin is in the bottom left
        # while the image coordinate frame origin is in the top left
        low_row = int((self.rows - self.y) - self.low_offset)
        high_row = int((self.rows - self.y) + self.high_offset)
        low_col = int(self.x - self.low_offset)
        high_col = int(self.x + self.high_offset)
        # ensure low row and col is non negative
        low_row = max(low_row, 0)
        low_col = max(low_col, 0)
        return self.terrain[low_row:high_row, low_col:high_col]

    @property
    def is_over_lot(self):
        """ Use the classifier to determine if the drone is currently over the lot.
        If the classifier's output is greater than or equal to .5, this is considered a
        positive prediction (over the lot). If self.flip_classes is set to True,
        this is reversed and an output less than or equal to .5 is considered to be
        over the lot.
        """
        if len(self.prior_predictions) >= NUM_PRIOR_PREDICTIONS:
            self.prior_predictions.pop(0)
        prediction = self.classifier.image_predict(self.camera_view)[0,0]
        if self.flip_classes:
            prediction = 1 - prediction
        self.prior_predictions.append(prediction)
        return (sum(self.prior_predictions) / NUM_PRIOR_PREDICTIONS) >= .5

    def update_state(self):
        if self.state == GO_STRAIGHT:
            if not self.is_over_lot:
                if self.turns_since_dir_change > TURNS_PER_DIRECTION:
                    self.turn_direction *= -1
                    self.turns_since_dir_change = 0
                self.turns_since_dir_change += 1
                self.state = FIND_LOT
                self.delta_theta = MAX_DELTA_THETA
        elif self.state == FIND_LOT:
            if self.is_over_lot:
                self.state = COUNTER_TURN
                self.counter_steps_to_go = self.rng.integers(low=MIN_COUNTER_TURN_STEPS, 
                    high=MAX_COUNTER_TURN_STEPS, endpoint=True)
                self.delta_theta = -MAX_COUNTER_TURN_DELTA_THETA
            else: # gradually widen the spiral
                self.delta_theta *= (1 - DELTA_THETA_DECAY_RATE)
        elif self.state == COUNTER_TURN:
            if self.counter_steps_to_go > 0:
                self.counter_steps_to_go -= 1
            else:
                self.state = GO_STRAIGHT
                self.delta_theta = 0
        else:
            raise ValueError('Invalid state: {}'.format(self.state))

    def step(self):
        """ Move forward one time step. Update state based on segmenter output
        and move according to new state.
        """
        self.update_state()
        self.move()
        self.path.append((self.x, self.y, self.theta))

    def run(self, iterations=10000):
        for i in range(iterations):
            self.step()
            if self.x < 0 or self.x >= self.cols or self.y < 0 or self.y >= self.rows:
                raise OutOfBoundsException(i)

    def _init_figure(self):
        fig, ax = plt.subplots()
        fig.tight_layout(pad=0)
        resolution = 500
        bg_img = cv2.resize(self.terrain, (resolution, resolution))
        ax.imshow(cv2.flip(bg_img, 0), origin='lower', extent=[1,6500, 1, 6500])
        ax.set_xlim([1,6500])
        ax.set_ylim([1,6500])

        return fig, ax

    def plot_path(self):
        fig, ax = self._init_figure()
        xs = [pose[0] for pose in self.path]
        ys = [pose[1] for pose in self.path]
        ax.plot(xs, ys, 'k')
        plt.show()

    def _animation_data_gen(self):
        xs = [pose[0] for pose in self.path]
        ys = [pose[1] for pose in self.path]
        interval = len(self.path) // NUM_ANIMATION_FRAMES
        for i in tqdm.tqdm(range(len(self.path) // interval)):
            yield (xs[:i * interval], ys[:i * interval])

    def _animation_step(self, frame, ax, line):
        xs, ys = frame
        line.set_data(xs, ys)
        return [line]

    def animate_path(self):
        fig, ax = self._init_figure()
        line,  = ax.plot(0, 0, 'k')
        fa = FuncAnimation(fig, fargs=[ax, line], blit=True,
            func=self._animation_step, frames=self._animation_data_gen(),
            save_count=NUM_ANIMATION_FRAMES
        )
        fa.save('animation2.mp4')


class OutOfBoundsException(Exception):
    def __init__(self, iteration):
        self.iteration = iteration


if __name__ == '__main__':
    params_path = 'params/en_12.55_md_dct_gs_5_nu_1_fe_rgb.params'
    features = params_path.split('_')[-1].split('.')[0]
    d_ctrl = DroneController(params_path, features)
    try:
        d_ctrl.run()
    except OutOfBoundsException as e:
        print('drone went out of bounds on iteration {}'.format(e.iteration))
    d_ctrl.animate_path()
    d_ctrl.plot_path()