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

WINDOW_NAME = 'Drone Path'

DELAY = .1 # sec

INTERVAL = 100


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
        self.last_updated = time.time()
        self.callbacks_since_last_recorded = 0

    def _pose_callback(self, msg):
        pos = msg.pose.position
        if self.callbacks_since_last_recorded >= INTERVAL:
            self.path_x.append(pos.x)
            self.path_y.append(pos.y)
            self.callbacks_since_last_recorded = 0
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
        print(type(image))
        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            exit()

    def _sleep(self):
        time_since_update = time.time() - self.last_updated
        sleep_duration = max(time_since_update, DELAY)
        time.sleep(sleep_duration)

    def run(self):
        print('initializing figure')
        self._init_figure()
        print('saving background for blitting')
        self._save_background_for_blitting()

        print('waiting for data to start arriving...')
        while not self.read_to_animate:
            time.sleep(.2)
        print('Starting animation. Press escape to exit and save to file.')
        self._display()
        time.sleep(2)
        self._create_path_line()
        while(1):
            self._update_path_line()
            self._restore_background()
            self._draw_path_line()
            self._display()
            self._sleep()
            print(len(self.path_x))






if __name__ == '__main__':
    dpp = DronePosPlotter('complex_parking_lot.png')
    dpp.run()