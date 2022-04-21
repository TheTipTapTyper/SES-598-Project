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

MAX_X_PIX = 6500
MIN_X_PIX = 1
MAX_Y_PIX = 6500
MIN_Y_PIX = 1

ANIMATION_RES = 500


WINDOW_NAME = 'Drone Path'

DELAY = .1 # sec

INTERVAL = 100


class DronePosPlotter:
    def __init__(self, terrain_fn, output_fn=None):
        self.running = False
        self.terrain_fn = terrain_fn
        self.output_fn = output_fn
        rospy.init_node('drone_pos_plotter', anonymous=True)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, 
            callback=self._pose_callback)
        self.path_x = []
        self.path_y = []
        self.ready_to_animate = False
        self.last_updated = time.time()
        self.callbacks_since_last_recorded = INTERVAL

    def _pose_callback(self, msg):
        pos = msg.pose.position
        if self.callbacks_since_last_recorded >= INTERVAL:
            x = np.interp(pos.x, (MIN_X_COORD, MAX_X_COORD), (MIN_X_PIX, MAX_X_PIX))
            y = np.interp(pos.y, (MIN_Y_COORD, MAX_Y_COORD), (MIN_Y_PIX, MAX_Y_PIX))
            self.path_x.append(x)
            self.path_y.append(y)
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
        print(type(image))
        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            exit()

    # def _sleep(self):
    #     time_since_update = time.time() - self.last_updated
    #     sleep_duration = max(time_since_update, DELAY)
    #     time.sleep(sleep_duration)

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
        #self._display()
        #time.sleep(2)
        self._create_path_line()
        self.running = True
        updates = 0
        while(1):
            self._step()
            updates += 1
            self.print('update #{} | {} points plotted'.format(updates, len(self.path_x)))

            # start = time.time()
            # self._update_path_line()
            # print('_update_path_line took {:.3f} sec'.format(time.time() - start))

            # start = time.time()
            # self._restore_background()
            # print('_restore_background took {:.3f} sec'.format(time.time() - start))

            # start = time.time()
            # self._draw_path_line()
            # print('_draw_path_line took {:.3f} sec'.format(time.time() - start))

            # start = time.time()
            # self._display()
            # print('_display took {:.3f} sec'.format(time.time() - start))

            # time.sleep(DELAY) #self._sleep()
            # print(len(self.path_x))






if __name__ == '__main__':
    dpp = DronePosPlotter('complex_parking_lot.png')
    dpp.run()