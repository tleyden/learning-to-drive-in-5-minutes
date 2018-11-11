#!/usr/bin/env python
# Copyright (c) 2018 Roma Sokolkov
# MIT License

import os
import gym
import numpy as np

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.common.policies import register_policy


from ddpg_with_vae import DDPGWithVAE as DDPG
from vae.controller import VAEController

# Registers donkey-vae-v0 gym env.
import donkey_gym_wrapper

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[32, 8],
                                           feature_extraction="mlp",
                                           layer_norm=True)

register_policy('CustomDDPGPolicy', CustomDDPGPolicy)


env = gym.make('donkey-vae-v0')

PATH_MODEL_VAE = "vae.json"
# Final filename will be PATH_MODEL_DDPG + ".pkl"
PATH_MODEL_DDPG = "ddpg"

# Initialize VAE model and add it to gym environment.
# VAE does image post processing to latent vector and
# buffers raw image for future optimization.
# z_size=512
vae = VAEController(z_size=512)
env.unwrapped.set_vae(vae)

# Run in test mode of trained models exist.
if os.path.exists(PATH_MODEL_DDPG + ".pkl") and \
   os.path.exists(PATH_MODEL_VAE):
    print("=================== Task: Testing ===================")
    ddpg = DDPG.load(PATH_MODEL_DDPG, env=env, policy=CustomDDPGPolicy)
    vae.load(PATH_MODEL_VAE)

    obs = env.reset()
    while True:
        action, _states = ddpg.predict(obs)
        # print(action)
        obs, reward, done, info = env.step(action)
        if done and not info.get('error_too_high'):
            env.reset()
        env.render()
# Run in training mode.
else:
    print("=================== Task: Training ===================")

    vae.load(PATH_MODEL_VAE)

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            # theta=float(0.6) * np.ones(n_actions),
            sigma=float(0.2) * np.ones(n_actions)
            )
    n_steps = 3000
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
                nb_train_steps=100,
                # normalize_observations=True,
                # normalize_returns=True
                )
    # ddpg = DDPG.load(PATH_MODEL_DDPG, env)
    ddpg.learn(total_timesteps=n_steps, vae=vae,
               skip_episodes=n_skip_ddpg, optimize_vae=False)
    # Finally save model files.
    ddpg.save(PATH_MODEL_DDPG)
    vae.save(PATH_MODEL_VAE)
