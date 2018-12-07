# Original Author: Roma Sokolkov (2018)
# Original repo: https://github.com/r7vme/learning-to-drive-in-a-day
# MIT License
# TODO: use speed set point instead of throttle
# Optimization is skipped during early episodes
import argparse
import os
import time

import gym
import numpy as np
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor


from ddpg_with_vae import DDPGWithVAE as DDPG
from sac_vae import SACWithVAE as SAC
from vae.controller import VAEController

# Registers donkey-vae-v0 gym env.
from donkey_gym_wrapper.env import DonkeyVAEEnv

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[32, 8],
                                           feature_extraction="mlp",
                                           layer_norm=True)

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           feature_extraction="mlp")

register_policy('CustomDDPGPolicy', CustomDDPGPolicy)

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='Force training')
parser.add_argument('--random-features', action='store_true', default=False,
                    help='Random Features')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--algo', help='RL algo', type=str, default='ddpg')
parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
parser.add_argument('-n', '--n-timesteps', help='Number of training timesteps', default=3000,
                    type=int)
parser.add_argument('--policy', help='Policy', default="LnMlpPolicy", type=str)
parser.add_argument('--n-stack', help='Number of frame stack', default=1,
                    type=int)
args = parser.parse_args()

if args.policy == "custom":
    if args.algo == 'ddpg':
        policy = CustomDDPGPolicy
    else:
        policy = CustomSACPolicy
else:
    if args.algo == 'ddpg':
        policy = LnMlpPolicy
    else:
        policy = MlpPolicy


set_global_seeds(args.seed)

MIN_THROTTLE = 0.2
Z_SIZE = 512
FRAME_SKIP = 2
TIMESTEPS = 0.05

# Initialize VAE model and add it to gym environment.
# VAE does image post processing to latent vector and
# buffers raw image for future optimization.
# z_size=512
vae = VAEController(z_size=Z_SIZE)

def make_env(seed=0):
    log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)
    def _init():
        env = DonkeyVAEEnv(level=0, time_step=TIMESTEPS, frame_skip=FRAME_SKIP,
                           z_size=Z_SIZE, vae=vae, const_throttle=None,
                           min_throttle=MIN_THROTTLE, max_throttle=0.4)
        env.seed(seed)
        env = Monitor(env, log_dir, allow_early_resets=True)
        return env
    return _init

env = DummyVecEnv([make_env(args.seed)])
env = VecFrameStack(env, n_stack=args.n_stack)

PATH_MODEL_VAE = "vae.json"
# env.venv.envs[0].unwrapped.set_vae(vae)

# Run in test mode of trained models exist.
if os.path.exists(args.algo + ".pkl") and \
   os.path.exists(PATH_MODEL_VAE) and not args.train:
    print("=================== Task: Testing ===================")
    if args.algo == 'ddpg':
        model = DDPG.load(args.algo, env=env, policy=policy)
    else:
        model = SAC.load(args.algo, env=env, policy=policy)
    vae.load(PATH_MODEL_VAE)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
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

    print("Using {}".format(args.algo))
    n_skip = 0 # n_episodes for gathering experience for VAE + random exploration

    if args.algo == 'ddpg':

        model = DDPG(policy,
                    env,
                    verbose=1,
                    batch_size=64,
                    clip_norm=0.005,
                    gamma=0.99,
                    param_noise=None,
                    action_noise=action_noise,
                    memory_limit=10000,
                    nb_train_steps=300,
                    normalize_observations=True,
                    normalize_returns=True
                    )
    else:
        model = SAC(policy,
                   env,
                   verbose=1,
                   gamma=0.99,
                   learning_starts=300,
                   buffer_size=10000,
                   batch_size=64,
                   learning_rate=3e-3,
                   # tau=0.05,
                   train_freq=2000,
                   gradient_steps=300,
                   ent_coef=0.01,
                   )
    # ddpg = DDPG.load('ddpg_fs8.pkl', policy=CustomDDPGPolicy, env=env)
    model.learn(total_timesteps=args.n_timesteps, vae=vae,
               skip_episodes=n_skip, optimize_vae=False,
               min_throttle=MIN_THROTTLE, log_interval=1)

    print("Training over, saving...")
    # Finally save model files.
    model.save(args.algo)

    vae.save(PATH_MODEL_VAE)
