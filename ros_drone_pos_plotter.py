import rospy
from geometry_msgs.msg import PoseStamped

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time


MAX_X_COORD = 150
MIN_X_COORD = -150
MAX_Y_COORD = 150
MIN_Y_COORD = -150

RATE = 10 # Hz

WINDOW_NAME = 'Drone Path'


class DronePosPlotter:
    def __init__(self, terrain_fn, output_fn=None):
        self.terrain_fn = terrain_fn
        self.output_fn = output_fn
        rospy.init_node('drone_pos_plotter', anonymous=True)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, 
            callback=self._pose_callback)
        self.path_x = []
        self.path_y = []
        self.read_to_animate = False
        self.rate = rospy.Rate(RATE)

    def _pose_callback(self, msg):
        pos = msg.pose.position
        self.path_x.append(pos.x)
        self.path_y.append(pos.y)
        self.read_to_animate = True

    def _init_figure(self):
        img = Image.open(self.terrain_fn)
        # remove alpha channel
        self.terrain = np.array(img)[:,:,:3]
        self.rows, self.cols = self.terrain.shape[:2]

        fig, ax = plt.subplots()
        fig.tight_layout(pad=0)
        resolution = 500
        bg_img = cv2.resize(self.terrain, (resolution, resolution))
        ax.imshow(cv2.flip(bg_img, 0), origin='lower', extent=[1,6500, 1, 6500])
        ax.set_xlim([1,6500])
        ax.set_ylim([1,6500])
        self.fig = fig
        self.ax = ax

    def _restore_background(self):
        self.fig.canvas.restore_region(self.background_img)

    def _save_background_for_blitting(self):
        self.fig.canvas.draw()
        self.background_img = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _create_path_line(self):
        self.path_line, = plt.plot()

    def _update_path_line(self):
        self.path_line.set_data()

    def _draw_path_line(self):
        self.ax.draw_artist(self.path_line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    @property
    def image(self):
        return np.ones((400,700, 3), dtype=np.uint8)

        return cv2.cvtColor(
            np.array(self.fig.canvas.renderer._renderer)[:,:,:3], 
            cv2.COLOR_RGB2BGR
        )

    def _display(self):
        cv2.imshow(WINDOW_NAME, self.image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            exit()

    def run(self):
        cv2.startWindowThread()
        cv2.namedWindow(WINDOW_NAME)

        print('initializing figure')
        self._init_figure()
        print('saving background for blitting')
        self._save_background_for_blitting()
        print('displaying saved image')
        print(self.image)
        print(self.image.shape)
        self._display()

        while(1):
            self._display()
            self.rate.sleep()

        exit()

        print('waiting for data to start arriving...')
        while not self.read_to_animate:
            time.sleep(.2)
        print('Starting animation. Press escape to exit and save to file.')

        self._create_path_line()

        while(1):
            self._update_path_line()
            self._restore_background()
            self._draw_path_line()
            self._display()
            self.rate.sleep()





if __name__ == '__main__':
    dpp = DronePosPlotter('complex_parking_lot.png')
    dpp.run()