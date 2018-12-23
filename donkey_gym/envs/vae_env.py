# Original author: Roma Sokolkov
# Hijacked donkey_gym wrapper with VAE.
#
# - Use Z vector as observation space.
# - Store raw images in VAE buffer.
import os

import numpy as np
import gym
from gym import spaces

from config import INPUT_DIM
from .donkey_env import DonkeyEnv
from .donkey_sim import DonkeyUnitySimContoller
from .donkey_proc import DonkeyUnityProcess
from config import MAX_STEERING


class DonkeyVAEEnv(DonkeyEnv):
    def __init__(self, level=0, frame_skip=2,
                 z_size=512, vae=None, const_throttle=None,
                 min_throttle=0.2, max_throttle=0.5,
                 max_cte_error=3.0, n_command_history=0):
        # super().__init__(level, time_step, frame_skip)
        self.z_size = z_size
        self.vae = vae

        self.const_throttle = const_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle

        # Save last n commands (throttle + steering)
        self.n_commands = 2
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_command_history = n_command_history

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
        self.viewer = DonkeyUnitySimContoller(level=level, port=port, max_cte_error=max_cte_error)

        if const_throttle is not None:
            # steering only
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING]),
                                           high=np.array([MAX_STEERING]),
                                           dtype=np.float32)
        else:
            # steering + throttle, action space must be symmetric
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
                                           high=np.array([MAX_STEERING, 1]), dtype=np.float32)

        if vae is None:
            assert n_command_history == 0, 'n_command_history not supported for images'
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=INPUT_DIM, dtype=np.uint8)
        else:
            # z latent vector
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.z_size + self.n_commands * n_command_history),
                                                dtype=np.float32)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.viewer.wait_until_loaded()

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.handler.send_exit_scene()

    def step(self, action):
        if self.const_throttle is not None:
            action = np.concatenate([action, [self.const_throttle]])
        else:
            # Convert from [-1, 1] to [0, 1]
            t = (action[1] + 1) / 2
            # Convert from [0, 1] to [min, max]
            action[1] = (1 - t) * self.min_throttle + self.max_throttle * t


        for repeat_idx in range(self.frame_skip):
            self.viewer.take_action(action, repeat_idx=repeat_idx)
            observation, reward, done, info = self._observe()

        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        return observation, reward, done, info

    def reset(self):
        self.viewer.reset()
        self.prev_error = None
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation, reward, done, info = self._observe()

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)
        return observation

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.viewer.handler.original_image
        return None

    def _observe(self):
        observation, reward, done, info = self.viewer.observe()
        # Learn from Pixels
        if self.vae is None:
            return observation, reward, done, info
        # Store image in VAE buffer.
        # self.vae.buffer_append(observation)
        return self.vae.encode(observation), reward, done, info

    def set_vae(self, vae):
        self.vae = vae
