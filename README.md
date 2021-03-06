# Learning to Drive Smoothly in Minutes

Learning to drive smoothly in minutes, using a reinforcement learning algorithm -- Soft Actor-Critic (SAC) -- and a Variational AutoEncoder (VAE) in the Donkey Car simulator.


Blog post on Medium: [link](https://medium.com/@araffin/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)

Video: [https://www.youtube.com/watch?v=iiuKh0yDyKE](https://www.youtube.com/watch?v=iiuKh0yDyKE)


Level-0          | Level-1
:-------------------------:|:-------------------------:
![result](content/smooth.gif)  | ![result](content/level1.gif)
[Download VAE](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) |  [Download VAE](https://drive.google.com/open?id=1hfQNAvVp2QmbmTLklWt2MxtAjrlisr2B)
[Download pretrained agent](https://drive.google.com/open?id=10Hgd5BKfn1AmmVdLlNcDll6yXqVkujoq) | [Download pretrained agent](https://drive.google.com/open?id=104tlsIrtOTVxJ1ZLoTpBDzK4-DRTA5et)

Note: the pretrained agents must be saved in `logs/sac/` folder (you need to pass `--exp-id 6` (index of the folder) to use the pretrained agent).


## Install SDSandbox unity simulator

1. Installing UnityHub and Unity 2018.3.14f1.
1. Git clone the [sdsandbox source](https://github.com/tawnkramer/sdsandbox/tree/donkey). 
1. `git checkout donkey` to switch to the donkey branch
1. Add a new project in Unity Hub and choose the sdsim directory
1. In Unity, go to File / Open Scene and choose the `road_generator.unity` file.


## Install Deep Reinforcement Learning controller deps + code (OSX)

1. Clone this repo
1. Create a virtualenv and activate it with `python3 -m venv env && source env/bin/activate`
1. Install OpenMPI via `brew install openmpi` which is needed for the `mpi4py` python dependency
1. Install dependencies via `pip install -r requirements.txt`

## Verify setup using pre-trained agent

1. Downlaod [pre-trained variational auto-encoder (VAE)](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b) and save in project directory root
1. Download [pretrained agent](https://drive.google.com/open?id=10Hgd5BKfn1AmmVdLlNcDll6yXqVkujoq) and unzip in `logs/sac` dir.  You should end up with a folder: `logs/sac/DonkeyVae-v0-level-0_6`
1. Run via:

    ```
    python enjoy.py --algo sac -vae vae-level-0-dim-32.pkl --exp-id 6 -n 2000
    ```
1. The script should be emitting: `Waiting for sim to start....` in a loop and it's waiting for the Unity simulation to connect to port `9090`
1. In the Unity window, hit the **Play** button near the top-center of the window, followed by the **Use NN Steering** button inside of the simulation window.
1. The car should start driving using the pretrained model, and it will restart whenever it goes off course.  To have it run longer, increase the `-n` parameter.

## Train

1. Downlaod [pre-trained variational auto-encoder (VAE)](https://drive.google.com/open?id=1n7FosFA0hALhuESf1j1yg-hERCnfVc4b)
1. Train a control policy for 5000 steps using Soft Actor-Critic (SAC)

    ```
    python train.py --algo sac -vae vae-level-0-dim-32.pkl -n 5000
    ```
1. The script should be emitting: `Waiting for sim to start....` in a loop and it's waiting for the Unity simulation to connect to port `9090`
1. In the Unity window, hit the **Play** button near the top-center of the window, followed by the **Use NN Steering** button inside of the simulation window.
1. The car should start driving using the pretrained model, and it will restart whenever it goes off course.  To have it run longer, increase the `-n` parameter.

The result files will be saved in a new directory under `logs/sac`, which can then be tested via the `enjoy.py` script.

## Train on a different level

To train on a different level:

1. [Download VAE level 1](https://drive.google.com/open?id=1hfQNAvVp2QmbmTLklWt2MxtAjrlisr2B) 
1. Change `LEVEL = 0` to `LEVEL = 1` in `config.py`
1. Re-run training

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

## Explore Latent Space

```
python -m vae.enjoy_latent -vae logs/level-0/vae-8.pkl
```

## Reproducing Results

To reproduce the results shown in the video, you have to check different values in `config.py`.

### Level 0

`config.py`:

```python
MAX_STEERING_DIFF = 0.15 # 0.1 for very smooth control, but it requires more steps
MAX_THROTTLE = 0.6 # MAX_THROTTLE = 0.5 is fine, but we can go faster
MAX_CTE_ERROR = 2.0 # only used in normal mode, set it to 10.0 when using teleoperation mode
LEVEL = 0
```

Train in normal mode (smooth control), it takes ~5-10 minutes:
```
python train.py --algo sac -n 8000 -vae logs/vae-level-0-dim-32.pkl
```

Train in normal mode (very smooth control with `MAX_STEERING_DIFF = 0.1`), it takes ~20 minutes:
```
python train.py --algo sac -n 20000 -vae logs/vae-level-0-dim-32.pkl
```

Train in teleoperation mode (`MAX_CTE_ERROR = 10.0`), it takes ~5-10 minutes:
```
python train.py --algo sac -n 8000 -vae logs/vae-level-0-dim-32.pkl --teleop
```

### Level 1

Note: only teleoperation mode is available for level 1

`config.py`:

```python
MAX_STEERING_DIFF = 0.15
MAX_THROTTLE = 0.5 # MAX_THROTTLE = 0.6 can work but it's harder to train due to the sharpest turn
LEVEL = 1
```

Train in teleoperation mode, it takes ~10 minutes:
```
python train.py --algo sac -n 15000 -vae logs/vae-level-1-dim-64.pkl --teleop
```

Note: although the size of the VAE is different between level 0 and 1, this is not an important factor.

## Record a Video of the on-board camera

You need a trained model. For instance, for recording 1000 steps with the last trained SAC agent:
```
python -m utils.record_video --algo sac --vae-path logs/level-0/vae-32-2.pkl -n 1000
```

## Citing the Project

To cite this repository in publications:

```
@misc{drive-smoothly-in-minutes,
  author = {Raffin, Antonin and Sokolkov, Roma},
  title = {Learning to Drive Smoothly in Minutes},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/araffin/learning-to-drive-in-5-minutes/}},
}
```

## Credits

Related Paper: ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf).

- [r7vme](https://github.com/r7vme/learning-to-drive-in-a-day) Author of the original implementation
- [Wayve.ai](https://wayve.ai) for idea and inspiration.
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for DDPG/SAC and PPO implementations.
- [RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo) for training/enjoy scripts.
- [S-RL Toolbox](https://github.com/araffin/robotics-rl-srl) for the data loader
- [Racing robot](https://github.com/sergionr2/RacingRobot) for the teleoperation
- [World Models Experiments](https://github.com/hardmaru/WorldModelsExperiments) for VAE implementation.
