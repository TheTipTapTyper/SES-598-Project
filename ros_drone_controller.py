"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 4/21/2022
Class: SES 598 Autonomous Exploration Systems
Project: Parking Lot Explorer

This file implements a drone controller class which uses a trained terrain
segmenter in order to stay over the top of a parking lot in order to locate
a car. The controller is a finite state machine (FSM) which moves in a straight
line while over the parking lot and begins to spiral when it finds itself over
the desert until it is once again over the lot. Once it does, it will counter
turn by a random degree for the sake of more even exploration/lot coverage.

This controller takes video and pose input from ros topics and publishes
velocity commands to the drone through another ros topic.
"""

import rospy
from rospy.numpy_msg import numpy_msg # https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String
import numpy as np
import random

from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String

import time

from detect_red_car import detect_red_obj
from terrain_cls import TerrainClassifier


WAYPOINT_THRESHOLD = .3
SET_MODE_SRV = '/mavros/set_mode'
CMD_ARMING_SRV = '/mavros/cmd/arming'
CUSTOM_MODE = 'OFFBOARD'
RATE = 10 # Hz

# FSM states
GO_STRAIGHT = 'GoStraight'      # Move forward until desert detected
FIND_LOT = 'FindLot'            # Turn with increasing radius until parking lot found
COUNTER_TURN = 'CounterTurn'    # Counter turn for a random duration
FINISHED = 'Finished'           # hover indefintely

TARGET_ALTITUDE = 5 # meters
TURNS_PER_DIRECTION = 5
MAX_VELOCITY = 3.5 # m/s
MAX_DELTA_THETA = 2.5 # degrees/step
DELTA_THETA_DECAY_RATE = 0.0065 # % per step
MAX_COUNTER_TURN_DURATION = 10 # sec
MIN_COUNTER_TURN_DURATION = 0 # sec
NUM_PRIOR_PREDICTIONS = 3
DISTANCE_THRESHOLD = 1 # meters
RED_COVERAGE_THRESHOLD = 0.05

MOVE_TYPE_VEL = 'vel'
MOVE_TYPE_POS = 'pos'


class DroneController:
    def __init__(self, t_classifier, lot_class=1):
        rospy.init_node('py_drone_ctrl', anonymous=True)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, 
            callback=self._pose_callback
        )
        rospy.Subscriber('/mavros/state', State, callback=self._state_callback)
        rospy.Subscriber('/uav_camera_down/image_raw', numpy_msg(SensorImage), 
            callback=self._image_callback
        )
        rospy.Subscriber('/mavros/local_', State, callback=self._state_callback)
        self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', 
            PoseStamped, queue_size=10
        )
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', 
            Twist, queue_size=10
        )
        self.status_pub = rospy.Publisher('/drone_controller/status', 
            String, queue_size=10
        )
        rospy.wait_for_service(SET_MODE_SRV)
        rospy.wait_for_service(CMD_ARMING_SRV)
        self.set_mode_service = rospy.ServiceProxy(SET_MODE_SRV, SetMode)
        self.cmd_arming_service = rospy.ServiceProxy(CMD_ARMING_SRV, CommandBool)
        self.x_pos = self.y_pos = self.z_pos = 0 # meters
        self.cmd_x = self.cmd_y = self.cmd_z = 0 # m/s
        self.heading = 0 # direction of travel in degrees
        self.delta_theta = 0 # deg/step
        self.is_ready = False
        self.rate = rospy.Rate(RATE)
        self.mode = None
        self.is_armed = False
        self.prior_predictions = []
        self.classifier = t_classifier
        self.flip_classes = not bool(lot_class)
        self.turns_since_dir_change = 0
        self.turn_direction = 1
        self.camera_view = None
        self.current_terrain = None
        self.final_x, self.final_y = None, None

    def ensure_correct_mode(self):
        if self.mode != CUSTOM_MODE:
            try:
                self.set_mode_service(1, CUSTOM_MODE)
            except rospy.ServiceException as e:
                print('/mavros/set_mode service call failed: ', e)
        if not self.is_armed:
            try:
                self.cmd_arming_service(True)
            except rospy.ServiceException as e:
                print('/mavros/cmd/arming service call failed: ', e)

    def _image_callback(self, msg):
        self.camera_view = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )

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
        result = (sum(self.prior_predictions) / len(self.prior_predictions)) >= .5
        if result:
            self.current_terrain = 'parking_lot'
        else:
            self.current_terrain = 'desert'
        return result

    def _pose_callback(self, msg):
        pos = msg.pose.position
        self.x_pos = pos.x
        self.y_pos = pos.y
        self.z_pos = pos.z

    def _state_callback(self, msg):
        self.mode = msg.mode
        self.is_armed = msg.armed

    def _publish_status(self):
        status = 'state: {}|terrain: {}|pos (xyz): {:.1f} {:.1f} {:.1f}|'
        status += 'heading: {:.1f} deg|delta_theta: {:.2f} deg/tick'
        status = status.format(self.state, self.current_terrain, self.x_pos, 
            self.y_pos, self.z_pos, self.heading, self.delta_theta
        )
        self.status_pub.publish(status)

    def update_state(self):
        if detect_red_obj(
            self.camera_view, coverage_threshold=RED_COVERAGE_THRESHOLD
        ) is not None:
            self.state = FINISHED
            self.final_x = self.x_pos
            self.final_y = self.y_pos
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
                self.counter_turn_duration = random.uniform(
                    MIN_COUNTER_TURN_DURATION, MAX_COUNTER_TURN_DURATION
                )
                self.counter_turn_start = time.time()
                self.delta_theta = -MAX_DELTA_THETA
            else: # gradually widen the spiral
                self.delta_theta *= (1 - DELTA_THETA_DECAY_RATE)
        elif self.state == COUNTER_TURN:
            if time.time() - self.counter_turn_start > self.counter_turn_duration:
                self.state = GO_STRAIGHT
                self.delta_theta = 0

    def move(self):
        if self.state == FINISHED: # hover
            cmd = PoseStamped()
            cmd.pose.position.x = self.final_x
            cmd.pose.position.y = self.final_y
            cmd.pose.position.z = TARGET_ALTITUDE
        else:
            cmd = Twist()
            z_cmd = 0
            z_diff = self.z_pos - TARGET_ALTITUDE
            if np.abs(z_diff) > DISTANCE_THRESHOLD:
                z_cmd = MAX_VELOCITY
                if z_diff > 0: # drone too high
                    z_cmd *= -1
            self.heading += self.turn_direction * self.delta_theta
            self.heading = self.heading % 360
            heading_rad = self.heading * np.pi / 180
            cmd.linear.x = np.cos(heading_rad) * MAX_VELOCITY
            cmd.linear.y = np.sin(heading_rad) * MAX_VELOCITY
            cmd.linear.z = z_cmd
            self.vel_pub.publish(cmd)

    def step(self):
        """ Move forward one time step. Update state based on segmenter output
        and move according to new state.
        """
        self.ensure_correct_mode()
        self.update_state()
        self.move()
        self._publish_status()
        print('{}: x: {:.2f} y: {:.2f} z: {:.2f} heading: {:.2f} d_theta: {:.2f}'.format(
            self.state, self.x_pos, self.y_pos, self.z_pos, self.heading, self.delta_theta
        ))

    def run(self):
        self.state = GO_STRAIGHT
        while self.camera_view is None:
            print('waiting for camera signal')
            time.sleep(.5)
        while(1):
            self.step()
            self.rate.sleep()


if __name__ == '__main__':
    params_path = 'params/en_12.55_md_dct_gs_5_nu_1_fe_rgb.params'
    features = params_path.split('_')[-1].split('.')[0]
    cls = TerrainClassifier(features=features)
    cls.load(params_path)
    d_ctrl = DroneController(cls)
    d_ctrl.run()