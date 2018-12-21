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
from config import INPUT_DIM, IMAGE_WIDTH, IMAGE_HEIGHT, ROI, THROTTLE_REWARD_WEIGHT,\
    MAX_THROTTLE, JERK_REWARD_WEIGHT, MIN_STEERING, MAX_STEERING, MAX_STEERING_DIFF



class DonkeyUnitySimContoller:

    def __init__(self, level, port=9090, max_cte_error=3.0):
        self.level = level
        self.verbose = False
        self.wait_time_for_obs = 0.1

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM

        self.address = ('0.0.0.0', port)

        self.handler = DonkeyUnitySimHandler(level, max_cte_error=max_cte_error)
        self.server = SimServer(self.address, self.handler)

        self.thread = Thread(target=asyncore.loop)
        self.thread.daemon = True
        self.thread.start()

    def close_connection(self):
        return self.server.handle_close()

    def wait_until_loaded(self):
        while not self.handler.loaded:
            print("Waiting for sim to start...")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        return self.handler.get_sensor_size()

    def take_action(self, action, repeat_idx=0):
        self.handler.take_action(action, repeat_idx)

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
    def __init__(self, level, max_cte_error=3.0):
        self.level_idx = level
        self.wait_time_for_obs = 0.1
        self.sock = None
        self.loaded = False
        self.verbose = False
        self.timer = FPSTimer(verbose=0)
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
        self.steering = None
        self.prev_steering = None

        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded}

    def on_connect(self, socket_handler):
        self.sock = socket_handler

    def on_disconnect(self):
        self.sock.close()
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

    def take_action(self, action, repeat_idx=0):
        if self.verbose:
            print("take_action")

        throttle = action[1]
        # Do not update prev_steering if using frame skip
        if repeat_idx == 0:
            self.prev_steering = self.steering
        self.steering = action[0]
        self.last_throttle = throttle
        self.current_step += 1

        self.send_control(self.steering, throttle)

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        info = {}

        self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self):
        return self.hit != "none" or math.fabs(self.cte) > self.max_cte_error

    # ------ RL interface ----------- #

    # Use velocity (m/s) as reward for every step,
    # except when episode done (failed).
    def calc_reward(self, done):
        if done:
            return -1
        # 1 per timesteps + velocity - jerk_penalty
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        jerk_penalty = 0
        if self.prev_steering is not None:
            steering_diff = (self.prev_steering - self.steering) / (MAX_STEERING - MIN_STEERING)

            if abs(steering_diff) > MAX_STEERING_DIFF:
                jerk_penalty = JERK_REWARD_WEIGHT * (steering_diff ** 2)
            else:
                jerk_penalty = 0
            # print(jerk_penalty, steering_diff)
        return 1 + throttle_reward - jerk_penalty

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Resize and crop image
        image = np.array(image)
        # Save original image for render
        self.original_image = np.copy(image)
        # Resize if using higher resolution images
        # image = cv2.resize(image, CAMERA_RESOLUTION)
        # Region of interest
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        self.image_array = image
        # Here resize is not useful for now (the image have already the right dimension)
        # self.image_array = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

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

    def send_exit_scene(self):
        msg = {'msg_type': 'exit_scene'}
        self.queue_message(msg)

    def queue_message(self, msg):
        if self.sock is None:
            if self.verbose:
                print('skipping:', msg)
            return

        if self.verbose:
            print('sending', msg)
        self.sock.queue_message(msg)
