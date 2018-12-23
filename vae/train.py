import argparse
import os

import numpy as np
from tqdm import tqdm
from stable_baselines.common import set_global_seeds

from vae.controller import VAEController
from .data_loader import DataLoader
from .model import ConvVAE

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image-folder', help='Path to a folder containing images for training',
                    default='', type=str)
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs/recorded_data/')
parser.add_argument('--z-size', help='Latent space', type=int, default=512)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-4)
parser.add_argument('--kl-tolerance', help='Learning rate', type=float, default=0.5)
parser.add_argument('--n-epochs', help='Number of epochs', type=int, default=10)
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = ConvVAE(z_size=args.z_size,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              kl_tolerance=args.kl_tolerance,
              is_training=True,
              reuse=False)

images = [im for im in os.listdir(args.folder) if im.endswith('.jpg')]
images = np.array(images)
n_samples = len(images)

print("{} images".format(n_samples))

# indices for all time steps where the episode continues
indices = np.arange(n_samples, dtype='int64')
np.random.shuffle(indices)

# split indices into minibatches. minibatchlist is a list of lists; each
# list is the id of the observation preserved through the training
minibatchlist = [np.array(sorted(indices[start_idx:start_idx + args.batch_size]))
                 for start_idx in range(0, len(indices) - args.batch_size + 1, args.batch_size)]

data_loader = DataLoader(minibatchlist, images, n_workers=2, folder=args.folder)

for epoch in range(args.n_epochs):
    pbar = tqdm(total=len(minibatchlist))
    for obs in data_loader:
        feed = {vae.x: obs}
        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
            vae.loss,
            vae.r_loss,
            vae.kl_loss,
            vae.global_step,
            vae.train_op
        ], feed)
        pbar.update(1)
    pbar.close()
    print("Epoch {:3}/{}".format(epoch + 1,args.n_epochs))
    print("VAE: optimization step", (train_step + 1), train_loss, r_loss, kl_loss)

save_path = "logs/vae-{}".format(args.z_size)
print("Saving to {}".format(save_path))
vae_controller = VAEController(z_size=args.z_size)
vae_controller.vae = vae
vae_controller.set_target_params()
vae.save(save_path)
