# Donkey_Gym

OpenAI gym environment for donkeycar simulator.

Reward function: scaled throttle + 1 for each timesteps and -1 when cross track error too high.
There is also a continuity penalty on steering to reduce jerk.
Episode ends when cross track error is too high or when the car hit something or reaches the end of the track.

# Credit

Original author: [Tawn Kramer]((https://github.com/tawnkramer/sdsandbox/tree/donkey/src/donkey_gym))
This version is based on the custom donkey gym from [Roma Sokolkov](https://github.com/r7vme) which includes a VAE env wrapper.
