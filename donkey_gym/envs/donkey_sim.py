# Original author: Tawn Kramer

import asyncore
import base64
import math
import time
from io import BytesIO
from threading import Thread

import cv2
import numpy as np
from PIL import Image

from donkey_gym.core.fps import FPSTimer
from donkey_gym.core.tcp_server import IMesgHandler, SimServer
from config import INPUT_DIM, IMAGE_WIDTH, IMAGE_HEIGHT, ROI



class DonkeyUnitySimContoller:

    def __init__(self, level, time_step=0.05, port=9090, max_cte_error=3.0):
        self.level = level
        self.time_step = time_step
        self.verbose = False
        self.wait_time_for_obs = 0.1

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM

        self.address = ('0.0.0.0', port)

        self.handler = DonkeyUnitySimHandler(level, time_step=time_step, max_cte_error=max_cte_error)
        self.server = SimServer(self.address, self.handler)

        self.thread = Thread(target=asyncore.loop)
        self.thread.daemon = True
        self.thread.start()

    def wait_until_loaded(self):
        while not self.handler.loaded:
            print("Waiting for sim to start...")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        return self.handler.observe()

    def quit(self):
        pass

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    FPS = 60.0

    def __init__(self, level, time_step=0.05, max_cte_error=3.0):
        self.level_idx = level
        self.time_step = time_step
        self.wait_time_for_obs = 0.1
        self.sock = None
        self.loaded = False
        self.verbose = False
        self.timer = FPSTimer(verbose=1)
        self.max_cte_error = max_cte_error

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.original_image = None
        self.last_obs = None
        self.last_throttle = 0.0
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.steering_angle  = 0.0
        self.current_step = 0
        self.speed = 0
        # self.error_too_high = False

        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded}

    def on_connect(self, socket_handler):
        self.sock = socket_handler

    def on_disconnect(self):
        self.sock = None

    def on_recv_message(self, message):
        if 'msg_type' not in message:
            print('Expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('Unknown message type', msg_type)

    # ------- Env interface ---------- #

    def reset(self):
        if self.verbose:
            print("reseting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.current_step = 0
        self.send_reset_car()
        self.send_control(0, 0)
        time.sleep(1.0)
        self.timer.reset()

    def get_sensor_size(self):
        return self.camera_img_size

    def take_action(self, action):
        if self.verbose:
            print("take_action")

        # Static throttle
        # throttle = 0.5
        throttle = action[1]
        self.last_throttle = throttle
        self.current_step += 1

        self.send_control(action[0], throttle)

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        # info = {'error_too_high': self.error_too_high}
        # info = {'original_image': self.original_image}
        info = {}

        self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self):
        # Workaround for big error at start.
        # if math.fabs(self.cte) > 2 * self.max_cte_error and self.current_step < 10:
        #     print("Too high error, ignoring {:.2f}".format(self.cte))
        #     self.error_too_high = True
        #     # self.send_get_scene_names()
        #     # self.send_load_scene("generated_road")
        #     # self.send_load_scene("warehouse")
        #     return False
        # self.error_too_high = False
        return self.hit != "none" or math.fabs(self.cte) > self.max_cte_error

    # ------ RL interface ----------- #

    # Use velocity (m/s) as reward for every step,
    # except when episode done (failed).
    def calc_reward(self, done):
        if done:
            return -1
        # 1 per timesteps + velocity
        # TODO: use real speed + jerk penalty
        velocity = self.last_throttle * (1.0 / self.FPS)
        return 1 + velocity

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Crop image and resize image
        r = ROI
        image = np.array(image)
        self.original_image = np.copy(image)
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Here resize is not useful for now (the image have already the right dimension)
        self.image_array = cv2.resize(np.array(image), (IMAGE_WIDTH, IMAGE_HEIGHT))

        # name of object we just hit. "none" if nothing.
        if self.hit == "none":
            self.hit = data["hit"]

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.steering_angle = data['steering_angle']
        self.speed = data["speed"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 3 scenes available now.
        try:
            self.cte = data["cte"]
        except KeyError:
            print("No CTE")
            pass

    def on_scene_selection_ready(self, _data):
        print("Scene Selection Ready")
        self.send_get_scene_names()

    def on_car_loaded(self, _data):
        if self.verbose:
            print("Car Loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        if data:
            names = data['scene_names']
            if self.verbose:
                print("SceneNames:", names)
            self.send_load_scene(names[self.level_idx])

    def send_control(self, steer, throttle):
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(), 'throttle': throttle.__str__(), 'brake': '0.0'}
        self.queue_message(msg)

    def send_reset_car(self):
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def send_get_scene_names(self):
        msg = {'msg_type': 'get_scene_names'}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        self.queue_message(msg)

    def queue_message(self, msg):
        if self.sock is None:
            if self.verbose:
                print('skipping:', msg)
            return

        if self.verbose:
            print('sending', msg)
        self.sock.queue_message(msg)
