import glob
import os
import time

import yaml
from stable_baselines import PPO2
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy

from algos import DDPG, SAC
from donkey_gym_wrapper.env import DonkeyVAEEnv
from vae.controller import VAEController

ALGOS = {
    # 'a2c': A2C,
    # 'acer': ACER,
    # 'acktr': ACKTR,
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2
}

MIN_THROTTLE = 0.6
MAX_THROTTLE = 0.8
Z_SIZE = 512
FRAME_SKIP = 2
TIMESTEPS = 0.05
MAX_CTE_ERROR = 3.5
LEVEL = 0
BASE_ENV = "DonkeyVae-v0"
ENV_ID = "DonkeyVae-v0-level-{}".format(LEVEL)


# ================== Custom Policies =================

class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                              layers=[16],
                                              feature_extraction="mlp")


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")


register_policy('CustomSACPolicy', CustomSACPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)


def load_vae(path=None, z_size=512):
    vae = VAEController(z_size=z_size)
    if path is not None:
        vae.load(path)
    return vae


def make_env(seed=0, log_dir=None, vae=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param seed: (int)
    :param log_dir: (str)
    :param vae: (str)
    """
    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(seed)
        env = DonkeyVAEEnv(level=LEVEL, time_step=TIMESTEPS, frame_skip=FRAME_SKIP,
                           z_size=Z_SIZE, vae=vae, const_throttle=None,
                           min_throttle=MIN_THROTTLE, max_throttle=MAX_THROTTLE,
                           max_cte_error=MAX_CTE_ERROR)
        env.seed(seed)
        env = Monitor(env, log_dir, allow_early_resets=True)
        return env

    return _init


def create_test_env(stats_path=None, seed=0,
                    log_dir='', hyperparams=None):
    """
    Create environment for testing a trained agent

    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :return: (gym.Env)
    """
    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = 'csv'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    vae_path = hyperparams['vae_path']
    if vae_path == '':
        vae_path = os.path.join(stats_path, 'vae.json')
    vae = None
    if stats_path is not None and os.path.isfile(vae_path):
        vae = load_vae(vae_path)

    env = DummyVecEnv([make_env(seed, log_dir, vae=vae)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams['normalize']:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])
            env.load_running_average(stats_path)

        n_stack = hyperparams.get('n_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_trained_models(log_folder):
    """
    :param log_folder: (str) Root log folder
    :return: (dict) Dict representing the trained agent
    """
    algos = os.listdir(log_folder)
    trained_models = {}
    for algo in algos:
        for env_id in glob.glob('{}/{}/*.pkl'.format(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split('/')[-1].split('.pkl')[0]
            trained_models['{}-{}'.format(algo, env_id)] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, 'config.yml')
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, 'config.yml'), 'r') as f:
                hyperparams = yaml.load(f)
            hyperparams['normalize'] = hyperparams.get('normalize', False)
        else:
            obs_rms_path = os.path.join(stats_path, 'obs_rms.pkl')
            hyperparams['normalize'] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams['normalize']:
            if isinstance(hyperparams['normalize'], str):
                normalize_kwargs = eval(hyperparams['normalize'])
            else:
                normalize_kwargs = {'norm_obs': hyperparams['normalize'], 'norm_reward': norm_reward}
            hyperparams['normalize_kwargs'] = normalize_kwargs
    return hyperparams, stats_path
