# Adapted from https://github.com/sergionr2/RacingRobot
# Author: Antonin Raffin
import time

import pygame
from pygame.locals import *

from config import MIN_STEERING, MAX_STEERING, MAX_THROTTLE, LEVEL
from donkey_gym.envs.vae_env import DonkeyVAEEnv
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


def update_screen(window, throttle, turn, is_recording, is_manual):
    clear(window)
    write_text(window, 'Throttle: {:.2f}, Angular: {:.2f}'.format(throttle, turn), 20, 0, FONT, WHITE)
    help_str = 'Use arrow keys to move, q or ESCAPE to exit.'
    write_text(window, help_str, 20, 50, SMALL_FONT)
    # help_2 = 'space key, k : force stop ---  anything else : stop smoothly'
    # write_text(window, help_2, 20, 100, SMALL_FONT)
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


def pygame_main(env):
    # Pygame require a window
    pygame.init()
    window = pygame.display.set_mode((800, 500), RESIZABLE)

    end = False
    is_recording = False
    is_manual = True

    control_throttle, control_turn = 0, 0
    update_screen(window, control_throttle, control_turn, is_recording, is_manual)

    last_time_pressed = {'space': 0, 'm': 0}

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
            _ = env.reset()

        if keys[K_l]:
            _ = env.reset()
            env.exit_scene()

        control_throttle, control_turn = control(x, theta, control_throttle, control_turn)
        # Send Orders
        angle_order = send_command(env, control_throttle, control_turn)

        update_screen(window, control_throttle, angle_order, is_recording, is_manual)

        for event in pygame.event.get():
            if event.type == QUIT or event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                end = True
        pygame.display.flip()
        # Limit FPS
        pygame.time.Clock().tick(1 / TELEOP_RATE)


def send_command(env, control_throttle, control_turn):
    """
    :param env: (Gym env)
    :param control_throttle: (float)
    :param control_turn: (float)
    """
    # Send Orders
    t = (control_turn + MAX_TURN) / (2 * MAX_TURN)
    angle_order = MIN_STEERING * t + MAX_STEERING * (1 - t)
    # TODO sen command
    action = [angle_order, control_throttle]
    _, _, _, _ = env.step(action)
    return angle_order


if __name__ == '__main__':
    env = DonkeyVAEEnv(level=LEVEL, frame_skip=1,
                       z_size=0, vae=None, const_throttle=None,
                       min_throttle=0, max_throttle=MAX_THROTTLE,
                       max_cte_error=10, n_command_history=0)
    env = Recorder(env, verbose=1)
    env.reset()
    try:
        pygame_main(env)
    except KeyboardInterrupt as e:
        pass
    finally:
        env.reset()
        env.exit_scene()
        time.sleep(0.5)
