# Learning to Drive Smoothly in 5 Minutes

Using Soft Actor-Critic (SAC) and a Variational AutoEncoder (VAE).


![result](content/smooth.gif)


Code that implements approach similar to described in ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf) paper.


Video with [real RC car](https://www.youtube.com/watch?v=6JUjDw9tfD4).


# Quick start

0. Download simulator
1. Install dependencies
2. (optional but recommended) Download pre-trained VAE
3. Train controller for 5000 steps

```
python train.py --algo sac -vae logs/vae.pkl -n 5000
```

4. Enjoy trained agent for 2000 steps

```
python enjoy.py --algo sac -vae logs/vae.pkl --exp-id 0 -n 2000
```

## Train vae

```
python -m vae.train --n-epochs 50 --verbose 0 --z-size 128 -f logs/recorded_data/
```

## Train in teleoparation mode

```
python train.py --algo sac -vae logs/vae.pkl -n 5000 --teleop
```

## Test in teleoparation mode

```
python -m teleop.teleop_client --algo sac -vae logs/vae.pkl --exp-id 0
```

# Credits

- [r7vme](https://github.com/r7vme/learning-to-drive-in-a-day) Author of the original implementation
- [Wayve.ai](https://wayve.ai) for idea and inspiration.
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for DDPG/SAC and PPO implementations.
- [RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo) for training/enjoy scripts.
- [S-RL Toolbox](https://github.com/araffin/robotics-rl-srl) for the data loader
- [Racing robot](https://github.com/sergionr2/RacingRobot) for the teleoperation
- [World Models Experiments](https://github.com/hardmaru/WorldModelsExperiments) for VAE implementation.
