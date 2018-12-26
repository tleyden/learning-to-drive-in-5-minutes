# Adapted from https://github.com/sergionr2/RacingRobot
# Author: Antonin Raffin
import os
import time
import argparse
from threading import Event, Thread

import pygame
import numpy as np
from pygame.locals import *
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines.bench import Monitor

from config import MIN_STEERING, MAX_STEERING, MIN_THROTTLE, MAX_THROTTLE, \
    LEVEL, N_COMMAND_HISTORY, TEST_FRAME_SKIP, ENV_ID
from donkey_gym.envs.vae_env import DonkeyVAEEnv
from utils.utils import ALGOS, get_latest_run_id, load_vae, linear_schedule
from .recorder import Recorder

UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)
STOP = (0, 0)
KEY_CODE_SPACE = 32

MAX_TURN = 1
# Smoothing constants
STEP_THROTTLE = 0.3
STEP_TURN = 0.4

TELEOP_RATE = 1 / 60  # 60 fps
GREEN = (72, 205, 40)
RED = (205, 39, 46)
GREY = (187, 179, 179)
BLACK = (36, 36, 36)
WHITE = (230, 230, 230)
ORANGE = (200, 110, 0)

moveBindingsGame = {
    K_UP: UP,
    K_LEFT: LEFT,
    K_RIGHT: RIGHT,
    K_DOWN: DOWN
}

pygame.font.init()
FONT = pygame.font.SysFont('Open Sans', 25)
SMALL_FONT = pygame.font.SysFont('Open Sans', 20)
KEY_MIN_DELAY = 0.4


def control(x, theta, control_throttle, control_turn):
    """
    Smooth control.

    :param x: (float)
    :param theta: (float)
    :param control_throttle: (float)
    :param control_turn: (float)
    :return: (float, float)
    """
    target_throttle = x
    target_turn = MAX_TURN * theta
    if target_throttle > control_throttle:
        control_throttle = min(target_throttle, control_throttle + STEP_THROTTLE)
    elif target_throttle < control_throttle:
        control_throttle = max(target_throttle, control_throttle - STEP_THROTTLE)
    else:
        control_throttle = target_throttle

    if target_turn > control_turn:
        control_turn = min(target_turn, control_turn + STEP_TURN)
    elif target_turn < control_turn:
        control_turn = max(target_turn, control_turn - STEP_TURN)
    else:
        control_turn = target_turn
    return control_throttle, control_turn


class TeleopEnv(object):
    def __init__(self, env, model=None, is_recording=False,
                 is_training=False, deterministic=True):
        super(TeleopEnv, self).__init__()
        self.env = env
        self.model = model
        self.need_reset = False
        self.is_manual = True
        self.is_recording = is_recording
        self.is_training = is_training
        self.current_obs = None
        self.exit_event = Event()
        self.done_event = Event()
        self.ready_event = Event()
        # For testing
        self.deterministic = deterministic
        self.window = None
        self.process = None
        self.action = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.donkey_env = None
        self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        self.process = Thread(target=self.main_loop)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def step(self, action):
        self.action = action
        self.current_obs, reward, done, info = self.env.step(action)
        # Overwrite done
        if self.done_event.is_set():
            done = True
            reward = -1
        else:
            done = False
        return self.current_obs, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def reset(self):
        # Disable reset after init
        if self.need_reset:
            self.need_reset = False
            return self.env.reset()
        else:
            return self.current_obs

    def wait_for_teleop_reset(self):
        self.ready_event.wait()
        if self.donkey_env.n_command_history > 0:
            if not self.observation_space.contains(self.current_obs):
                self.current_obs = np.concatenate((self.current_obs,
                                                   np.zeros_like(self.donkey_env.command_history)), axis=-1)
        return self.reset()

    def exit(self):
        self.env.reset()
        self.donkey_env.exit_scene()

    def wait(self):
        self.process.join()

    def main_loop(self):
        # Pygame require a window
        pygame.init()
        self.window = pygame.display.set_mode((800, 500), RESIZABLE)

        end = False
        self.is_recording = False
        self.is_manual = True

        control_throttle, control_turn = 0, 0
        action = [control_turn, control_throttle]
        self.update_screen(action)

        donkey_env = self.env
        # Unwrap env
        if isinstance(donkey_env, Recorder):
            donkey_env = donkey_env.env
        while isinstance(donkey_env, VecNormalize) or isinstance(donkey_env, VecFrameStack):
            donkey_env = donkey_env.venv

        if isinstance(donkey_env, DummyVecEnv):
            donkey_env = donkey_env.envs[0]
        if isinstance(donkey_env, Monitor):
            donkey_env = donkey_env.env

        assert isinstance(donkey_env, DonkeyVAEEnv), print(donkey_env)
        self.donkey_env = donkey_env

        last_time_pressed = {'space': 0, 'm': 0, 't': 0}
        self.current_obs = self.reset()

        while not end:
            x, theta = 0, 0
            keys = pygame.key.get_pressed()
            for keycode in moveBindingsGame.keys():
                if keys[keycode]:
                    x_tmp, th_tmp = moveBindingsGame[keycode]
                    x += x_tmp
                    theta += th_tmp

            if keys[K_SPACE] and (time.time() - last_time_pressed['space']) > KEY_MIN_DELAY:
                self.is_recording = not self.is_recording
                if isinstance(self.env, Recorder):
                    self.env.toggle_recording()
                # avoid multiple key press
                last_time_pressed['space'] = time.time()

            if keys[K_m] and (time.time() - last_time_pressed['m']) > KEY_MIN_DELAY:
                self.is_manual = not self.is_manual
                # avoid multiple key press
                last_time_pressed['m'] = time.time()
                if self.is_training:
                    if self.is_manual:
                        # Stop training
                        self.ready_event.clear()
                        self.done_event.set()
                    else:
                        # Start training
                        self.done_event.clear()
                        self.ready_event.set()

            if keys[K_t] and (time.time() - last_time_pressed['t']) > KEY_MIN_DELAY:
                self.is_training = not self.is_training
                # avoid multiple key press
                last_time_pressed['t'] = time.time()

            if keys[K_r]:
                self.current_obs = self.env.reset()

            if keys[K_l]:
                self.env.reset()
                self.donkey_env.exit_scene()
                self.need_reset = True

            # Smooth control for teleoperation
            control_throttle, control_turn = control(x, theta, control_throttle, control_turn)
            # Send Orders
            if self.model is None or self.is_manual:
                t = (control_turn + MAX_TURN) / (2 * MAX_TURN)
                angle_order = MIN_STEERING * t + MAX_STEERING * (1 - t)
                self.action = [angle_order, control_throttle]
            elif self.model is not None and not self.is_training:
                if not self.observation_space.contains(self.current_obs):
                    self.current_obs = np.concatenate((self.current_obs,
                                                       np.zeros_like(self.donkey_env.command_history)), axis=-1)
                self.action, _ = self.model.predict(self.current_obs, deterministic=self.deterministic)

            if not (self.is_training and not self.is_manual):
                if self.is_manual:
                    donkey_env.viewer.take_action(self.action)
                    self.current_obs, _, _, _ = donkey_env._observe()
                else:
                    self.current_obs, _, _, _ = self.env.step(self.action)

            self.update_screen(self.action)

            for event in pygame.event.get():
                if event.type == QUIT or event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                    end = True
            pygame.display.flip()
            # Limit FPS
            pygame.time.Clock().tick(1 / TELEOP_RATE)
        self.ready_event.set()
        self.exit_event.set()

    def write_text(self, text, x, y, font, color=GREY):
        text = str(text)
        text = font.render(text, True, color)
        self.window.blit(text, (x, y))

    def clear(self):
        self.window.fill((0, 0, 0))

    def update_screen(self, action):
        self.clear()
        turn, throttle = action
        self.write_text('Throttle: {:.2f}, Angular: {:.2f}'.format(throttle, turn), 20, 0, FONT, WHITE)
        help_str = 'Use arrow keys to move, q or ESCAPE to exit.'
        self.write_text(help_str, 20, 50, SMALL_FONT)
        help_2 = 'space key: toggle recording -- m: change mode -- r: reset -- l: reset track'
        self.write_text(help_2, 20, 100, SMALL_FONT)
        self.write_text('Recording Status:', 20, 150, SMALL_FONT, WHITE)

        if self.is_recording:
            text, text_color = 'RECORDING', RED
        else:
            text, text_color = 'NOT RECORDING', GREEN

        self.write_text(text, 200, 150, SMALL_FONT, text_color)

        self.write_text('Mode:', 20, 200, SMALL_FONT, WHITE)

        if self.is_manual:
            text, text_color = 'MANUAL', GREEN
        else:
            text, text_color = 'AUTONOMOUS', ORANGE

        self.write_text(text, 200, 200, SMALL_FONT, text_color)

        self.write_text('Training Status:', 20, 250, SMALL_FONT, WHITE)

        if self.is_training:
            text, text_color = 'TRAINING', RED
        else:
            text, text_color = 'TESTING', GREEN

        self.write_text(text, 200, 250, SMALL_FONT, text_color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--record-folder', help='Record folder, where images are saved', type=str,
                        default='logs/recorded_data/')
    parser.add_argument('--algo', help='RL Algorithm', default='',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (-1: no exp folder, 0: latest)', default=0,
                        type=int)
    parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
    args = parser.parse_args()

    algo = args.algo
    folder = args.folder
    model = None
    vae = None

    if algo != '':
        if args.exp_id == 0:
            args.exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)
            print('Loading latest experiment, id={}'.format(args.exp_id))

        # Sanity checks
        if args.exp_id > 0:
            log_path = os.path.join(folder, algo, '{}_{}'.format(ENV_ID, args.exp_id))
        else:
            log_path = os.path.join(folder, algo)

        model_path = "{}/{}.pkl".format(log_path, ENV_ID)

        assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
        assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, ENV_ID, model_path)
        model = ALGOS[algo].load(model_path)

    if args.vae_path != '':
        print("Loading VAE ...")
        vae = load_vae(args.vae_path)

    if vae is None:
        N_COMMAND_HISTORY = 0

    env = DonkeyVAEEnv(level=LEVEL, frame_skip=TEST_FRAME_SKIP, vae=vae, const_throttle=None, min_throttle=MIN_THROTTLE,
                       max_throttle=MAX_THROTTLE, max_cte_error=10, n_command_history=N_COMMAND_HISTORY)
    env = Recorder(env, folder=args.record_folder, verbose=1)
    try:
        env = TeleopEnv(env, model=model)
        # model = ALGOS['sac']('CustomSACPolicy', env, verbose=1,
        #                      buffer_size=10000, gradient_steps=300,
        #                      ent_coef=0.2, batch_size=64, train_freq=2500,
        #                      learning_rate=linear_schedule(3e-3))
        # env.model = model
        # model.learn(10000, log_interval=1)
        # model.save("logs/sac/{}".format(ENV_ID))
        # env.model = model
        env.wait()
    except KeyboardInterrupt as e:
        pass
    finally:
        env.exit()
        time.sleep(0.5)
