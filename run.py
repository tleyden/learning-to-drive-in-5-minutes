#!/usr/bin/env python
# Copyright (c) 2018 Roma Sokolkov
# MIT License
import argparse
import os
import gym
import numpy as np

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines.common import set_global_seeds


from ddpg_with_vae import DDPGWithVAE as DDPG
from vae.controller import VAEController

# Registers donkey-vae-v0 gym env.
from donkey_gym_wrapper.env import DonkeyVAEEnv

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[32, 8],
                                           feature_extraction="mlp",
                                           layer_norm=True)

register_policy('CustomDDPGPolicy', CustomDDPGPolicy)

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='Force training')
parser.add_argument('--random-features', action='store_true', default=False,
                    help='Random Features')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
parser.add_argument('-n', '--n-timesteps', help='Number of training timesteps', default=3000,
                    type=int)
parser.add_argument('--n-stack', help='Number of frame stack', default=1,
                    type=int)
args = parser.parse_args()


set_global_seeds(args.seed)

# Initialize VAE model and add it to gym environment.
# VAE does image post processing to latent vector and
# buffers raw image for future optimization.
# z_size=512
vae = VAEController(z_size=512)

def make_env(seed=0):
    def _init():
        env = DonkeyVAEEnv(level=0, time_step=0.05, frame_skip=2,
                           z_size=512, vae=vae, const_throttle=None,
                           min_throttle=0.2, max_throttle=0.4)
        env.seed(seed)
        return env
    return _init

env = DummyVecEnv([make_env(args.seed)])
env = VecFrameStack(env, n_stack=args.n_stack)

PATH_MODEL_VAE = "vae.json"
# Final filename will be PATH_MODEL_DDPG + ".pkl"
PATH_MODEL_DDPG = "ddpg"

# env.venv.envs[0].unwrapped.set_vae(vae)

# Run in test mode of trained models exist.
if os.path.exists(PATH_MODEL_DDPG + ".pkl") and \
   os.path.exists(PATH_MODEL_VAE) and not args.train:
    print("=================== Task: Testing ===================")
    ddpg = DDPG.load(PATH_MODEL_DDPG, env=env, policy=CustomDDPGPolicy)
    vae.load(PATH_MODEL_VAE)

    obs = env.reset()
    while True:
        action, _states = ddpg.predict(obs)
        # print(action)
        # Throttle
        # print("{:.2f}".format(action[0, 1]))
        obs, reward, done, info = env.step(action)
        if done and not info[0].get('error_too_high'):
            env.reset()
        env.render()
# Run in training mode.
else:
    print("=================== Task: Training ===================")

    if not args.random_features:
        print("Loading VAE...")
        vae.load(PATH_MODEL_VAE)

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            # theta=float(0.6) * np.ones(n_actions),
            sigma=float(0.2) * np.ones(n_actions)
            )

    n_skip_ddpg = 0 # n_episodes for gathering experience for VAE

    ddpg = DDPG('CustomDDPGPolicy',
                env,
                verbose=1,
                batch_size=64,
                clip_norm=None,
                gamma=0.99,
                param_noise=None,
                action_noise=action_noise,
                memory_limit=10000,
                nb_train_steps=400,
                # normalize_observations=True,
                # normalize_returns=True
                )
    # ddpg = DDPG.load('ddpg_fs8.pkl', policy=CustomDDPGPolicy, env=env)
    ddpg.learn(total_timesteps=args.n_timesteps, vae=vae,
               skip_episodes=n_skip_ddpg, optimize_vae=False)

    print("Training over, saving...")
    # Finally save model files.
    ddpg.save(PATH_MODEL_DDPG)
    vae.save(PATH_MODEL_VAE)
