import rospy
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from sensor_msgs.msg import Image
import math
import numpy

from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler, euler_from_quaternion

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

TURNS_PER_DIRECTION = 5
MAX_DELTA_THETA = .15 # rad/s
MAX_COUNTER_TURN_DURATION = 5 # sec
NUM_PRIOR_PREDICTIONS = 3

class DroneController:
    def __init__(self):
        rospy.init_node('py_drone_ctrl', anonymous=True)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, 
            callback=self._pose_callback)
        rospy.Subscriber('/mavros/state', State, callback=self._state_callback)
        #rospy.Subscriber('/mavros/local_', State, callback=self._state_callback)
        rospy.Subscriber('/uav_camera_down/image_raw', Image, callback=self._image_callback)
        #self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        rospy.wait_for_service(SET_MODE_SRV)
        rospy.wait_for_service(CMD_ARMING_SRV)
        self.set_mode_service = rospy.ServiceProxy(SET_MODE_SRV, SetMode)
        self.cmd_arming_service = rospy.ServiceProxy(CMD_ARMING_SRV, CommandBool)
        # self.des_pose = PoseStamped()
        # self.position = (0, 0, 0)
        # self.velocity = (0, 0, 0)
        # self.angular_velocity = (0, 0, 0)
        # self.angles = (0, 0, 0)
        self.is_ready = False
        self.rate = rospy.Rate(RATE)
        self.mode = None
        self.is_armed = False

    def _image_callback(self, msg):
        self.image = msg

    @property
    def camera_view(self):
        """ current view of the downward facing drone camera.
        """
        pass

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
        quat = pose.pose.orientation
        self.position = (pos.x, pos.y, pos.z)
        angles = (euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]))
        self.angles = tuple(rad2deg(angle) for angle in angles)

    def _state_callback(self, msg):
        self.mode = msg.mode
        self.is_armed = msg.armed
        #print('mode: {} armed: {}'.format(msg.mode, msg.armed))

    # def _velocity_callback(self, msg):
    #     print(msg)

    def update_state(self):
        pass
        # if self.state == GO_STRAIGHT:
        #     if not self.is_over_lot:
        #         if self.turns_since_dir_change > TURNS_PER_DIRECTION:
        #             self.turn_direction *= -1
        #             self.turns_since_dir_change = 0
        #         self.turns_since_dir_change += 1
        #         self.state = FIND_LOT
        #         self.delta_theta = MAX_DELTA_THETA
        # elif self.state == FIND_LOT:
        #     if self.is_over_lot:
        #         self.state = COUNTER_TURN
        #         self.counter_steps_to_go = self.rng.integers(low=MIN_COUNTER_TURN_STEPS, 
        #             high=MAX_COUNTER_TURN_STEPS, endpoint=True)
        #         self.delta_theta = -MAX_COUNTER_TURN_DELTA_THETA
        #     else: # gradually widen the spiral
        #         self.delta_theta *= (1 - DELTA_THETA_DECAY_RATE)
        # elif self.state == COUNTER_TURN:
        #     if self.counter_steps_to_go > 0:
        #         self.counter_steps_to_go -= 1
        #     else:
        #         self.state = GO_STRAIGHT
        #         self.delta_theta = 0
        # else:
        #     raise ValueError('Invalid state: {}'.format(self.state))

    def step(self):
        """ Move forward one time step. Update state based on segmenter output
        and move according to new state.
        """
        self.update_state()
        self.move()
        self.path.append((self.x, self.y, self.theta))

    def run(self):
        pass

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