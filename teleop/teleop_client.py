# Adapted from https://github.com/sergionr2/RacingRobot
# Author: Antonin Raffin
import os
import time
import argparse

import pygame
from pygame.locals import *

from config import MIN_STEERING, MAX_STEERING, MIN_THROTTLE, MAX_THROTTLE,\
    LEVEL, Z_SIZE, N_COMMAND_HISTORY, TEST_FRAME_SKIP, ENV_ID
from donkey_gym.envs.vae_env import DonkeyVAEEnv
from utils.utils import ALGOS, get_latest_run_id, load_vae
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
STEP_TURN = 0.5

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
KEY_MIN_DELAY = 0.2


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


def write_text(screen, text, x, y, font, color=GREY):
    text = str(text)
    text = font.render(text, True, color)
    screen.blit(text, (x, y))


def clear(window):
    window.fill((0, 0, 0))


def update_screen(window, action, is_recording, is_manual):
    clear(window)
    turn, throttle = action
    write_text(window, 'Throttle: {:.2f}, Angular: {:.2f}'.format(throttle, turn), 20, 0, FONT, WHITE)
    help_str = 'Use arrow keys to move, q or ESCAPE to exit.'
    write_text(window, help_str, 20, 50, SMALL_FONT)
    help_2 = 'space key: toggle recording -- m: change mode -- r: reset -- l: reset track'
    write_text(window, help_2, 20, 100, SMALL_FONT)
    write_text(window, 'Status:', 20, 150, SMALL_FONT, WHITE)

    if is_recording:
        text, text_color = 'RECORDING', RED
    else:
        text, text_color = 'NOT RECORDING', GREEN

    write_text(window, text, 100, 150, SMALL_FONT, text_color)

    write_text(window, 'Mode:', 20, 200, SMALL_FONT, WHITE)
    if is_manual:
        text, text_color = 'MANUAL', GREEN
    else:
        text, text_color = 'AUTONOMOUS', ORANGE

    write_text(window, text, 100, 200, SMALL_FONT, text_color)


def pygame_main(env, model=None):
    # Pygame require a window
    pygame.init()
    window = pygame.display.set_mode((800, 500), RESIZABLE)

    end = False
    is_recording = False
    is_manual = True

    control_throttle, control_turn = 0, 0
    action = [control_turn, control_throttle]
    update_screen(window, action, is_recording, is_manual)

    last_time_pressed = {'space': 0, 'm': 0}
    obs = env.reset()

    while not end:
        x, theta = 0, 0
        keys = pygame.key.get_pressed()
        for keycode in moveBindingsGame.keys():
            if keys[keycode]:
                x_tmp, th_tmp = moveBindingsGame[keycode]
                x += x_tmp
                theta += th_tmp

        if keys[K_SPACE] and (time.time() - last_time_pressed['space']) > KEY_MIN_DELAY:
            is_recording = not is_recording
            env.toggle_recording()
            # avoid multiple key press
            last_time_pressed['space'] = time.time()

        if keys[K_m] and (time.time() - last_time_pressed['m']) > KEY_MIN_DELAY:
            is_manual = not is_manual
            # avoid multiple key press
            last_time_pressed['m'] = time.time()

        if keys[K_r]:
            obs = env.reset()

        if keys[K_l]:
            obs = env.reset()
            env.exit_scene()

        control_throttle, control_turn = control(x, theta, control_throttle, control_turn)
        # Send Orders
        action, obs = send_command(env, control_throttle, control_turn, obs, is_manual, model)

        update_screen(window, action, is_recording, is_manual)

        for event in pygame.event.get():
            if event.type == QUIT or event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                end = True
        pygame.display.flip()
        # Limit FPS
        pygame.time.Clock().tick(1 / TELEOP_RATE)


def send_command(env, control_throttle, control_turn, obs, is_manual, model=None):
    """
    :param env: (Gym env)
    :param control_throttle: (float)
    :param control_turn: (float)
    :param obs: (np.ndarray)
    :param is_manual: (bool)
    :param model: (RL object)
    """
    if model is None or is_manual:
        # Send Orders
        t = (control_turn + MAX_TURN) / (2 * MAX_TURN)
        angle_order = MIN_STEERING * t + MAX_STEERING * (1 - t)
        action = [angle_order, control_throttle]
    else:
        action, _ = model.predict(obs)

    obs, _, _, _ = env.step(action)
    return action, obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
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
            vae = load_vae(args.vae_path, z_size=Z_SIZE)

    if vae is None:
        N_COMMAND_HISTORY = 0

    env = DonkeyVAEEnv(level=LEVEL, frame_skip=TEST_FRAME_SKIP,
                       z_size=Z_SIZE, vae=vae, const_throttle=None,
                       min_throttle=MIN_THROTTLE, max_throttle=MAX_THROTTLE,
                       max_cte_error=10, n_command_history=N_COMMAND_HISTORY)
    env = Recorder(env, verbose=1)
    try:
        pygame_main(env, model=model)
    except KeyboardInterrupt as e:
        pass
    finally:
        env.reset()
        env.exit_scene()
        time.sleep(0.5)
