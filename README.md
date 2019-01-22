# Learning to Drive Smoothly in Minutes

Learning to drive smoothly in minutes, using a reinforcement learning algorithm -- Soft Actor-Critic (SAC) -- and a Variational AutoEncoder (VAE) in the Donkey Car simulator.

Level-0          | Level-1
:-------------------------:|:-------------------------:
![result](content/smooth.gif)  | ![result](content/level1.gif)
[Download VAE](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) |  [Download VAE](https://drive.google.com/open?id=1hfQNAvVp2QmbmTLklWt2MxtAjrlisr2B)
[Download pretrained agent] | [Download pretrained agent](https://drive.google.com/open?id=1cSW1AoTgBoMtjCKCYCYJ4sJjYCeDeqhO)




# Quick Start

0. Download simulator [here](https://drive.google.com/open?id=1h2VfpGHlZetL5RAPZ79bhDRkvlfuB4Wb) or build it from [source](https://github.com/tawnkramer/sdsandbox/tree/donkey)
1. Install dependencies (cf requirements.txt)
2. (optional but recommended) Download pre-trained VAE: [VAE Level 0](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) [VAE Level 1](https://drive.google.com/open?id=1hfQNAvVp2QmbmTLklWt2MxtAjrlisr2B)
3. Train a control policy for 5000 steps using Soft Actor-Critic (SAC)

```
python train.py --algo sac -vae path-to-vae.pkl -n 5000
```

4. Enjoy trained agent for 2000 steps

```
python enjoy.py --algo sac -vae path-to-vae.pkl --exp-id 0 -n 2000
```

To train on a different level, you need to change `LEVEL = 0` to `LEVEL = 1` in `config.py`

## Train the Variational AutoEncoder (VAE)

0. Collect images using the teleoperation mode:

```
python -m teleop.teleop_client --record-folder path-to-record/folder/
```

1. Train a VAE:
```
python -m vae.train --n-epochs 50 --verbose 0 --z-size 64 -f path-to-record/folder/
```

## Train in Teleoparation Mode

```
python train.py --algo sac -vae logs/vae.pkl -n 5000 --teleop
```

## Test in Teleoparation Mode

```
python -m teleop.teleop_client --algo sac -vae logs/vae.pkl --exp-id 0
```

Related Paper: ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf).

# Credits

- [r7vme](https://github.com/r7vme/learning-to-drive-in-a-day) Author of the original implementation
- [Wayve.ai](https://wayve.ai) for idea and inspiration.
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for DDPG/SAC and PPO implementations.
- [RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo) for training/enjoy scripts.
- [S-RL Toolbox](https://github.com/araffin/robotics-rl-srl) for the data loader
- [Racing robot](https://github.com/sergionr2/RacingRobot) for the teleoperation
- [World Models Experiments](https://github.com/hardmaru/WorldModelsExperiments) for VAE implementation.
