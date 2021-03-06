"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 4/21/2022
Class: SES 598 Autonomous Exploration Systems
Project: Parking Lot Explorer

This module implements a ros node which serves as a visualization tool for the
ros_drone_controller node.
"""

import rospy
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from detect_red_car import detect_red_obj

MAX_X_COORD = 150
MIN_X_COORD = -150
MAX_Y_COORD = 150
MIN_Y_COORD = -150

ANIMATION_RES = 500
WINDOW_NAME = 'Drone Path'
DELAY = .1 # sec
INTERVAL = 25
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = .6
FONT_COLOR = (0, 0, 0)
THICKNESS = 1
TEXT_X = 20
TEXT_Y = 20


class DronePosPlotter:
    def __init__(self, terrain_fn, output_fn=None):
        self.running = False
        self.terrain_fn = terrain_fn
        self.output_fn = output_fn
        rospy.init_node('drone_pos_plotter', anonymous=True)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, 
            callback=self._pose_callback
        )
        rospy.Subscriber('/uav_camera_down/image_raw', numpy_msg(SensorImage), 
            callback=self._image_callback
        )
        rospy.Subscriber('/drone_controller/status', String, 
            callback=self._status_callback
        )
        self.path = None
        self.last_updated = time.time()
        self.callbacks_since_last_recorded = INTERVAL
        self.camera_view = None
        self.fig_image_shape = None
        self.d_ctrl_status = None

    def _status_callback(self, msg):
        self.d_ctrl_status = msg.data

    def _image_callback(self, msg):
        if self.fig_image_shape is not None:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )
            h, w, _ = self.fig_image_shape
            image = cv2.resize(image, (w, h))
            detect_red_obj(image)
            self.camera_view = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _pose_callback(self, msg):
        pos = msg.pose.position
        if self.callbacks_since_last_recorded >= INTERVAL:
            if self.path is None:
                self.path = np.array([pos.x, pos.y])
            else:
                self.path = np.vstack([
                    self.path,
                    (pos.x, pos.y)
                ])
            self.callbacks_since_last_recorded = 0
        self.callbacks_since_last_recorded += 1

    def _init_figure(self):
        img = Image.open(self.terrain_fn)
        # remove alpha channel
        self.terrain = np.array(img)[:,:,:3]
        self.rows, self.cols = self.terrain.shape[:2]
        fig, ax = plt.subplots()
        fig.tight_layout(pad=0)
        bg_img = cv2.resize(self.terrain, (ANIMATION_RES, ANIMATION_RES))
        ax.imshow(cv2.flip(bg_img, 0), origin='lower', extent=[
            MIN_X_COORD, MAX_X_COORD, MIN_Y_COORD, MAX_Y_COORD
        ])
        ax.set_xlim([MIN_X_COORD, MAX_X_COORD])
        ax.set_ylim([MIN_Y_COORD, MAX_Y_COORD])
        self.fig = fig
        self.ax = ax

    def _restore_background(self):
        self.fig.canvas.restore_region(self.background_img)

    def _save_background_for_blitting(self):
        self.fig.canvas.draw()
        self.background_img = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _create_path_line(self):
        self.path_line, = plt.plot(self.path[:,0], self.path[:,1], 'k')

    def _update_path_line(self):
        self.path_line.set_data(self.path[:,0], self.path[:,1])

    def _draw_path_line(self):
        self.ax.draw_artist(self.path_line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def _add_status_text(self, image):
        if self.d_ctrl_status is not None:
            status_elements = self.d_ctrl_status.split('|')
            for idx, element in enumerate(status_elements):
                image = cv2.putText(
                    img=image, 
                    text=element, 
                    org=(TEXT_X, TEXT_Y + idx*TEXT_Y),
                    fontFace=FONT, 
                    fontScale=FONT_SCALE, 
                    color=FONT_COLOR,
                    thickness=THICKNESS
                )
        return image

    def _display(self):
        fig_image = cv2.cvtColor(
            np.array(self.fig.canvas.renderer._renderer)[:,:,:3], 
            cv2.COLOR_RGB2BGR
        )
        self.fig_image_shape = fig_image.shape
        full_image = fig_image
        if self.camera_view is not None:
            cam_image = self._add_status_text(self.camera_view)
            full_image = np.vstack([fig_image, cam_image])
        cv2.imshow(WINDOW_NAME, full_image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            exit()

    def _step(self):
        self._update_path_line()
        self._restore_background()
        self._draw_path_line()
        self._display()

    def run(self):
        print('initializing figure')
        self._init_figure()
        print('saving background for blitting')
        self._save_background_for_blitting()

        print('waiting for data to start arriving...')
        while self.path is None:
            time.sleep(.2)
        print('Starting animation. Press escape to exit.')
        self._create_path_line()
        self.running = True
        updates = 0
        while(1):
            self._step()
            updates += 1
            time.sleep(.2)


if __name__ == '__main__':
    dpp = DronePosPlotter('complex_parking_lot.png')
    dpp.run()