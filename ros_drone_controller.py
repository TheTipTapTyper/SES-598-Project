import rospy
from rospy.numpy_msg import numpy_msg # https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from sensor_msgs.msg import Image
import math
import numpy as np
from numpy.random import default_rng

from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String
#from tf.transformations import quaternion_from_euler, euler_from_quaternion

from itertools import cycle
import time

from detect_red_car import detect_red_obj

WAYPOINT_THRESHOLD = .3
SET_MODE_SRV = '/mavros/set_mode'
CMD_ARMING_SRV = '/mavros/cmd/arming'
CUSTOM_MODE = 'OFFBOARD'
RATE = 10 # Hz

# FSM states
INIT = 'Init'                   # move to initial position
GO_STRAIGHT = 'GoStraight'      # Move forward until desert detected
FIND_LOT = 'FindLot'            # Turn with increasing radius until parking lot found
COUNTER_TURN = 'CounterTurn'    # Counter turn for a random duration
SEEK_CAR = 'SeekCar'            # Move towards car until in center of view
FINISHED = 'Finished'           # hover indefintely

TARGET_ALTITUDE = 15 # meters
TURNS_PER_DIRECTION = 5
MAX_VELOCITY = 1 # m/s
MAX_DELTA_THETA = .15 # rad/s
MAX_COUNTER_TURN_DURATION = 5 # sec
MIN_COUNTER_TURN_DURATION = 2 # sec
NUM_PRIOR_PREDICTIONS = 3
DISTANCE_THRESHOLD = 1 # meters
DELTA_THETA_DECAY_RATE = 0.01 # % per step

class DroneController:
    def __init__(self):
        rospy.init_node('py_drone_ctrl', anonymous=True)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, 
            callback=self._pose_callback)
        rospy.Subscriber('/mavros/state', State, callback=self._state_callback)
        rospy.Subscriber('/uav_camera_down/image_raw', numpy_msg(Image), callback=self._image_callback)
        rospy.Subscriber('/mavros/local_', State, callback=self._state_callback)
        #self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        rospy.wait_for_service(SET_MODE_SRV)
        rospy.wait_for_service(CMD_ARMING_SRV)
        self.set_mode_service = rospy.ServiceProxy(SET_MODE_SRV, SetMode)
        self.cmd_arming_service = rospy.ServiceProxy(CMD_ARMING_SRV, CommandBool)
        self.x_pos = self.y_pos = self.z_pos = 0 # meters
        self.x_vel = self.y_vel = self.z_vel = 0 # m/s
        self.delta_theta = 0 # rad/s
        self.is_ready = False
        self.rate = rospy.Rate(RATE)
        self.mode = None
        self.is_armed = False
        self.rng = default_rng()

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
        self.camera_view = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

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

    def _pose_callback(self, pose):
        pos = pose.pose.position
        self.x_pos = pos.x
        self.y_pos = pos.y
        self.z_pos = pos.z
        # quat = pose.pose.orientation
        # angles = (euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]))
        # self.angles = tuple(rad2deg(angle) for angle in angles)

    def _state_callback(self, msg):
        self.mode = msg.mode
        self.is_armed = msg.armed
        #print('mode: {} armed: {}'.format(msg.mode, msg.armed))

    def update_state(self):
        if self.state == INIT:
            if TARGET_ALTITUDE - self.y_pos < DISTANCE_THRESHOLD:
                self.state = GO_STRAIGHT
                self.y_vel = 0
                self.x_vel = MAX_VELOCITY
        elif self.state == GO_STRAIGHT:
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
                self.counter_turn_duration = self.rng.uniform(
                    MIN_COUNTER_TURN_DURATION, MAX_COUNTER_TURN_DURATION
                )
                self.delta_theta = -MAX_DELTA_THETA
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

    def move(self):
        cmd = Twist()
        cmd.linear.x = self.x_vel
        cmd.linear.y = self.y_vel
        cmd.linear.z = self.z_vel
        cmd.angular.z = self.delta_theta
        cmd.angular.x = cmd.angular.y = 0
        self.vel_pub.publish(cmd)

    def step(self):
        """ Move forward one time step. Update state based on segmenter output
        and move according to new state.
        """
        self.ensure_correct_mode()
        self.update_state()
        self.move()
        print('{}: x: {:.2f} y: {:.2f} z: {:.2f} xv: {:.2f} yv: {:.2f} zv: {:.2f} d_theta: {:.2f}'.format(
            self.state, self.x_pos, self.y_pos, self.z_pos, self.x_vel, self.y_vel, self.z_vel, self.delta_theta
        ))

    def run(self):
        self.state = INIT
        while(1):
            self.step()
            self.rate.sleep()



def distance(pos1, pos2):
    total = 0
    for coord1, coord2 in zip(pos1, pos2):
        diff = coord1 - coord2
        total += diff ** 2
    return math.sqrt(total)

def rad2deg(angle):
    return angle * 180 / math.pi

def deg2rad(angle):
    return angle * math.pi / 180


if __name__ == '__main__':
    d_ctrl = DroneController()
    d_ctrl.run()