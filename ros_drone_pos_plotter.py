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


MAX_X_COORD = 150
MIN_X_COORD = -150
MAX_Y_COORD = 150
MIN_Y_COORD = -150

MAX_X_PIX = 6500
MIN_X_PIX = 1
MAX_Y_PIX = 6500
MIN_Y_PIX = 1

ANIMATION_RES = 500
WINDOW_NAME = 'Drone Path'
DELAY = .1 # sec
INTERVAL = 25
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (0, 255, 0)
TEXT_COORDS = (20,100)


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
        self.path_x = []
        self.path_y = []
        self.ready_to_animate = False
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
            if self.d_ctrl_status is not None:
                image = cv2.putText(
                    img=image, 
                    text=self.d_ctrl_status, 
                    org=TEXT_COORDS,
                    fontFace=FONT, 
                    fontScale=FONT_SCALE, 
                    color=FONT_COLOR
                )
            self.camera_view = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _pose_callback(self, msg):
        pos = msg.pose.position
        if self.callbacks_since_last_recorded >= INTERVAL:
            x = np.interp(pos.x, (MIN_X_COORD, MAX_X_COORD), (MIN_X_PIX, MAX_X_PIX))
            y = np.interp(pos.y, (MIN_Y_COORD, MAX_Y_COORD), (MIN_Y_PIX, MAX_Y_PIX))
            self.path_x.append(x)
            self.path_y.append(y)
            print(x, y)
            self.callbacks_since_last_recorded = 0
            self.ready_to_animate = True
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
            MIN_X_PIX, MAX_X_PIX, MIN_Y_PIX, MAX_Y_PIX
        ])
        ax.set_xlim([MIN_X_PIX, MAX_X_PIX])
        ax.set_ylim([MIN_Y_PIX, MAX_Y_PIX])
        self.fig = fig
        self.ax = ax

    def _restore_background(self):
        self.fig.canvas.restore_region(self.background_img)

    def _save_background_for_blitting(self):
        self.fig.canvas.draw()
        self.background_img = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _create_path_line(self):
        self.path_line, = plt.plot(self.path_x, self.path_y, 'k')

    def _update_path_line(self):
        self.path_line.set_data(self.path_x, self.path_y)

    def _draw_path_line(self):
        self.ax.draw_artist(self.path_line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def _display(self):
        image = cv2.cvtColor(
            np.array(self.fig.canvas.renderer._renderer)[:,:,:3], 
            cv2.COLOR_RGB2BGR
        )
        self.fig_image_shape = image.shape
        if self.camera_view is not None:
            image = np.vstack([image, self.camera_view])
        cv2.imshow(WINDOW_NAME, image)
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
        while not self.ready_to_animate:
            time.sleep(.2)
        print('Starting animation. Press escape to exit and save to file.')
        self._create_path_line()
        self.running = True
        updates = 0
        while(1):
            self._step()
            updates += 1
            print('update #{} | {} points plotted'.format(updates, len(self.path_x)))
            time.sleep(.2)






if __name__ == '__main__':
    dpp = DronePosPlotter('complex_parking_lot.png')
    dpp.run()