# Learning to Drive in 5 Minutes

Using Soft Actor-Critic (SAC) and a Variational AutoEncoder (VAE).


![result](content/example-after-3000-steps.gif)

Video with [real RC car](https://www.youtube.com/watch?v=6JUjDw9tfD4).

Code that implements approach similar to described in ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf) paper.

Missing parts:
- Prioritized Experience Replay in DDPG. Right now we randomly sample.
- Params well tuning to drive more smoothly.

# Quick start

0. Download simulator
1. Install dependencies
2. (optional but recommended) Download pre-trained VAE
3. Train controller

```
python train.py --algo sac -vae logs/vae.json
```

# Under the hood

Script does the following:
- Initialize Donkey OpenAI gym environment.
- Initialize VAE controller with random weights.
- If no pretrained models found, run in train mode. Otherwise just load weights from files and run test.
- Initialize DDPG controller.
- Learning function will collect the data by running 10 episodes w/o DDPG optimization, then after every episode DDPG optimization happens. VAE optimized after every episode.
- After 3000 steps training will be finished and weights params will be saved to files.

# Troubleshooting

## Disable VAE optimization

Implementation is still very very raw and needs fine tuning, so to get quick results i recommend to run full session and then reuse `vae.json` (or use [pretrained](https://drive.google.com/open?id=16WYkH7goKnJM52ke1KAzs5vozGiuKPqu)) in new training session by adding `vae.load(PATH_MODEL_VAE)` before `ddpg.learn` and commenting out `vae.optimize()` in `ddpg_with_vae.py`. This will allow to train DDPG very quickly even on CPU machine.

## Visualize what car sees

Following code can be used to decode VAE's Z latent vector and save to image.

```
arr = vae.decode(obs)
arr = np.round(arr).astype(np.uint8)
arr = arr.reshape(80, 160, 3)
# pip install Pillow
img = PIL.Image.fromarray(arr)
img.save('decoded_img.jpg')
```

## Try pretrained models

Just to make sure that environment was setup correctly try [pretrained models](https://drive.google.com/open?id=16WYkH7goKnJM52ke1KAzs5vozGiuKPqu). Place `ddpg.pkl` and `vae.json` into the root directory. You should see similar to the GIF above.

# Credits

- [r7vme](https://github.com/r7vme/learning-to-drive-in-a-day) Author of the original implementation
- [Wayve.ai](https://wayve.ai) for idea and inspiration.
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for DDPG/SAC and PPO implementations.
- [RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo) for training/enjoy scripts.
- [S-RL Toolbox](https://github.com/araffin/robotics-rl-srl) for the data loader
- [World Models Experiments](https://github.com/hardmaru/WorldModelsExperiments) for VAE implementation.
