# Copyright (c) 2018 Roma Sokolkov
# MIT License

"""
Hijacked donkey_gym wrapper with VAE.

- Use Z vector as observation space.
- Store raw images in VAE buffer.

Problem that DDPG already well implemented in stable-baselines
and VAE integration will require full reimplementation of DDPG
codebase. Instead we hijack VAE into gym environment.
"""

import os

import numpy as np
import gym
from gym import spaces
from donkey_gym.envs.donkey_env import DonkeyEnv
from donkey_gym.envs.donkey_sim import DonkeyUnitySimContoller
from donkey_gym.envs.donkey_proc import DonkeyUnityProcess


class DonkeyVAEEnv(DonkeyEnv):
    def __init__(self, level=0, time_step=0.05, frame_skip=2,
                 z_size=512, vae=None, const_throttle=None,
                 min_throttle=0.2, max_throttle=0.5):
        # super().__init__(level, time_step, frame_skip)
        self.z_size = z_size
        self.vae = vae
        # PID
        self.k_p = 0.4 # P factor
        self.k_d = 0.1
        self.prev_error = None

        self.const_throttle = const_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle

        print("starting DonkeyGym env")
        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        try:
            exe_path = os.environ['DONKEY_SIM_PATH']
        except KeyError:
            print("Missing DONKEY_SIM_PATH environment var. Using defaults")
            # you must start the executable on your own
            exe_path = "self_start"

        try:
            port = int(os.environ['DONKEY_SIM_PORT'])
        except KeyError:
            print("Missing DONKEY_SIM_PORT environment var. Using defaults")
            port = 9091

        try:
            headless = os.environ['DONKEY_SIM_HEADLESS'] == '1'
        except KeyError:
            print("Missing DONKEY_SIM_HEADLESS environment var. Using defaults")
            headless = False

        self.proc.start(exe_path, headless=headless, port=port)

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(level=level, time_step=time_step, port=port)

        if const_throttle is not None:
            # steering only
            self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        else:
            # steering + throttle, action space must be symmetric
            self.action_space = spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1, 1]), dtype=np.float32)

        # z latent vector
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.z_size), dtype=np.float32)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.viewer.wait_until_loaded()

    def step(self, action):
        # error = current_theta - action
        # error_d = 0.0
        # if self.prev_error is not None:
        #     error_d = (error - self.prev_error)
        #
        # command = self.k_p * error + self.k_d * error_d
        # self.prev_error = error
        # print("action=", action[0])
        if self.const_throttle is not None:
            action = np.concatenate([action, [self.const_throttle]])
        else:
            # Convert from [-1, 1] to [0, 1]
            t = (action[1] + 1) / 2
            # Convert from [0, 1] to [min, max]
            action[1] = (1 - t) * self. min_throttle + self.max_throttle * t

        # print(action)

        for _ in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self._observe()
        return observation, reward, done, info

    def reset(self):
        self.viewer.reset()
        self.prev_error = None
        observation, reward, done, info = self._observe()
        return observation

    def _observe(self):
        observation, reward, done, info = self.viewer.observe()
        # Solves chicken-egg problem as gym calls reset before we call set_vae.
        if not hasattr(self, "vae"):
            return np.zeros(self.z_size), reward, done, info
        # Store image in VAE buffer.
        self.vae.buffer_append(observation)
        return self.vae.encode(observation), reward, done, info

    def set_vae(self, vae):
        self.vae = vae
