import time
from collections import deque

import numpy as np
import tensorflow as tf
from stable_baselines import SAC, logger
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.ppo2.ppo2 import safe_mean
from stable_baselines import logger


class SACWithVAE(SAC):
    """docstring for SACWithVAE."""
    def learn(self, total_timesteps, callback=None, seed=None, vae=None,
              skip_episodes=0, optimize_vae=False, min_throttle=0, writer=None,
              log_interval=1, print_freq=100):
        self._setup_learn(seed)

        start_time = time.time()
        episode_rewards = [0.0]
        obs = self.env.reset()
        self.episode_reward = np.zeros((1,))
        ep_info_buf = deque(maxlen=100)
        ep_len = 0
        n_updates = 0
        infos_values = []
        mb_infos_vals = []

        for step in range(total_timesteps):
            if callback is not None:
                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                if callback(locals(), globals()) is False:
                    break

            # Before training starts, randomly sample actions
            # from a uniform distribution for better exploration.
            # Afterwards, use the learned policy.
            if step < self.learning_starts:
                action = self.env.action_space.sample()
                # No need to rescale when sampling random action
                rescaled_action = action
            else:
                action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                # Rescale from [-1, 1] to the correct bounds
                rescaled_action = action * np.abs(self.action_space.low)

            assert action.shape == self.env.action_space.shape

            new_obs, reward, done, info = self.env.step(rescaled_action)
            ep_len += 1

            if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
                print("{} steps".format(ep_len))

            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            # Retrieve reward and episode length if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_info_buf.extend([maybe_ep_info])

            if writer is not None:
                ep_reward = np.array([reward]).reshape((1, -1))
                ep_done = np.array([done]).reshape((1, -1))
                self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                  ep_done, writer, step)

            if ep_len > self.train_freq:
                mb_infos_vals = []
                print("Additional training")
                for grad_step in range(self.gradient_steps):
                    if step < self.batch_size or step < self.learning_starts:
                        break
                    mb_infos_vals.append(self._train_step(step, writer))
                    if (step + grad_step) % self.target_update_interval == 0:
                        # Update target network
                        self.sess.run(self.target_update_op)
                self.env.reset()
                done = True

            episode_rewards[-1] += reward
            if done:
                mb_infos_vals = []
                if not isinstance(self.env, VecEnv):
                    obs = self.env.reset()

                # Prevent the car from moving during training
                # self.env.step([0.0, -min_throttle])

                print("Episode finished. Reward: {:.2f}".format(episode_rewards[-1]))
                episode_rewards.append(0.0)
                ep_len = 0
                for grad_step in range(self.gradient_steps):
                    if step < self.batch_size or step < self.learning_starts:
                        break
                    n_updates += 1
                    # Update policy and critics (q functions)
                    mb_infos_vals.append(self._train_step(step, writer))

                    if (step + grad_step) % self.target_update_interval == 0:
                        # Update target network
                        self.sess.run(self.target_update_op)


            # Log losses and entropy, useful for monitor training
            if len(mb_infos_vals) > 0:
                infos_values = np.mean(mb_infos_vals, axis=0)

            if len(episode_rewards[-101:-1]) == 0:
                mean_reward = -np.inf
            else:
                mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

            num_episodes = len(episode_rewards)
            if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                fps = int(step / (time.time() - start_time))
                logger.logkv("episodes", num_episodes)
                logger.logkv("mean 100 episode reward", mean_reward)
                logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                logger.logkv("n_updates", n_updates)
                logger.logkv("fps", fps)
                logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                if len(infos_values) > 0:
                    for (name, val) in zip(self.infos_names, infos_values):
                        logger.logkv(name, val)
                logger.logkv("total timesteps", step)
                logger.dumpkvs()
                # Reset infos:
                infos_values = []
        return self
