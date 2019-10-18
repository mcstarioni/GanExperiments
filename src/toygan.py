import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.autograd
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import animatplot as amp

import src.datasets as toys
import src.utils as utils

from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#%%

# dataset = toys.GaussianGrid(10000, rows=5, cols=5)
# dataset = toys.GaussianSpiral(10000, rotations=2)
dataset = toys.GaussianCircle(10000, clusters=15)
dataset.visualize()

#%%

def spectral_block(fin, fout, wrapper: utils.SpectralWrapper, activations):
    return nn.Sequential(wrapper.wrap(nn.Linear(fin, fout)), activations)


#%%

# Build Generator
z_size = 2
x_size = int(np.prod(dataset.sample_size))

generator_conf = [z_size, 200, 200, 100, 50]
g_wrapper = utils.SpectralWrapper(False)  # Use SN or Not
# g_post_linear = lambda fout: nn.Sequential(nn.ReLU(), nn.LayerNorm(fout))  # Activations and normalizations
g_post_linear = lambda fout: nn.Sequential(nn.ReLU())  # Activations and normalizations


# Build neural network
# Last layer has no activations or normalizations
g_block_list = [spectral_block(fin, fout, g_wrapper, g_post_linear(fout))
                for fin, fout in zip(generator_conf[:-1], generator_conf[1:])]
g_last_layer = g_wrapper.wrap(nn.Linear(generator_conf[-1], x_size))


g_model = nn.Sequential(*g_block_list, g_last_layer)
g_optim = torch.optim.Adam(g_model.parameters(), lr=0.001, betas=(0.5, 0.9))
print(g_model, utils.count_parameters(g_model))


#%%

# Build Discriminator
discriminator_conf = [x_size, 150, 100, 70, 50]
d_wrapper = utils.SpectralWrapper(True)
# d_post_linear = lambda fout: nn.Sequential(nn.ReLU(), nn.LayerNorm(fout))
d_post_linear = lambda fout: nn.Sequential(nn.ReLU())


# Build neural network
# Last layer has no activations or normalizations
d_block_list = [spectral_block(fin, fout, d_wrapper, d_post_linear(fout))
                for fin, fout in zip(discriminator_conf[:-1], discriminator_conf[1:])]
d_last_layer = d_wrapper.wrap(nn.Linear(discriminator_conf[-1], 1))


d_model = nn.Sequential(*d_block_list, d_last_layer)
d_optim = torch.optim.Adam(d_model.parameters(), lr=0.001, betas=(0.5, 0.99))
print(d_model, utils.count_parameters(d_model))


#%%

# initialize plotting variables
n_epochs = 1000
swap_iters = 2
log_iters = 15
n_iters = 0

z_xx, z_yy = utils.mesh(-3, 3, 20)
x_xx, x_yy = utils.mesh(-25, 25, 50)

z_initial_shape = z_xx.shape
x_initial_shape = x_xx.shape

# Evenly sa
z_grid = torch.from_numpy(np.reshape(np.stack([z_xx, z_yy], axis=-1), (-1, 2))).float()
x_grid = torch.from_numpy(np.reshape(np.stack([x_xx, x_yy], axis=-1), (-1, 2))).float()

print(z_grid.shape, x_grid.shape)

loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
ones = torch.ones(loader.batch_size, 1)
zeros = torch.zeros(loader.batch_size, 1)
disc_output_list = []
gen_output_list = []


#%%

# g_val stands for validity
for i in range(n_epochs):
    for true, in loader:
        if n_iters % swap_iters == 0:

            g_optim.zero_grad()

            z = torch.randn((true.shape[0], z_size))
            fake = g_model(z)
            fake_val = d_model(fake)
            true_val = d_model(true)

            # Relativistic GAN from https://ajolicoeur.wordpress.com/relativisticgan/
            # fake_avg = torch.mean(fake_val).expand_as(fake_val)#, keepdim=True)
            # true_avg = torch.mean(true_val).expand_as(true_val)#, keepdim=True)
            # print(true_avg.shape, true_val.shape)
            #
            # true_loss = torch.mean(true_val - fake_avg)
            # fake_loss = torch.mean(true_avg - fake_val)
            # g_loss = true_loss + fake_loss

            # Original GAN
            # g_loss = F.binary_cross_entropy_with_logits(fake_val, ones)

            # LSGAN
            g_loss = F.mse_loss(fake_val, ones)

            # WGAN-div
            # g_loss = -torch.mean(fake_val) # Maximize Fake val

            g_loss.backward()
            g_optim.step()
            print(n_iters, "G:", g_loss.item())

        else:
            d_optim.zero_grad()

            z = torch.randn((true.shape[0], z_size))
            fake = g_model(z)

            fake_val = d_model(fake)
            true_val = d_model(true)

            gp = utils.gradient_penalty(true, fake, d_model, k=5, p=10)

            # Relativistic GAN from https://ajolicoeur.wordpress.com/relativisticgan/
            # fake_avg = torch.mean(fake_val).expand_as(fake_val)#, keepdim=True)
            # true_avg = torch.mean(true_val).expand_as(true_val)#, keepdim=True)
            #
            # true_loss = torch.mean(fake_avg - true_val)
            # fake_loss = torch.mean(fake_val - true_avg)

            # Original GAN
            # true_loss = F.binary_cross_entropy_with_logits(true_val, ones)
            # fake_loss = F.binary_cross_entropy_with_logits(fake_val, zeros)

            # LSGAN
            true_loss = F.mse_loss(true_val, ones)
            fake_loss = F.mse_loss(fake_val, zeros)

            # WGAN-div
            # fake_loss = torch.mean(fake_val) # Minimize Fake val
            # true_loss = -torch.mean(true_val) # Maximize True val

            d_loss = true_loss + fake_loss + gp

            d_loss.backward()
            d_optim.step()
            print(n_iters, "D:", d_loss.item())

        if n_iters % log_iters == 0:
            g_model.eval()
            gen_output_list.append(g_model(z_grid).clone().detach().numpy())
            disc_output_list.append(d_model(x_grid).clone().detach().numpy())

        n_iters += 1


# %%

# Plot results
plt.cla()
z = torch.randn((250, z_size))
pred = g_model(z).detach().numpy()
print(pred.shape)
dataset.visualize(c="blue")
plt.scatter(pred[:, 0], pred[:, 1], c="red")


# %%

x = d_model(x_grid).detach().numpy()
print(x_grid.shape, x.shape)
plt.cla()
plt.contourf(x_xx, x_yy, np.reshape(x, x_initial_shape))


# %%

# Plot training
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

start_iter = 0
stop_iter = n_iters
steps = 1

start = start_iter//log_iters
stop = stop_iter//log_iters
slice = np.s_[start:stop:steps]

x_array = np.array(disc_output_list)
x_array = np.reshape(x_array, (-1, *x_initial_shape))[slice]
x_array = utils.std_normalize_samplewise_ndarray(x_array)

block2 = amp.blocks.Pcolormesh(x_xx, x_yy, x_array, ax=ax, t_axis=0, cmap="viridis")
dataset.visualize(ax=ax, c="blue")
ax.set_aspect('equal')

z_array = np.array(gen_output_list)
block1 = amp.blocks.Scatter(z_array[slice, :, 0], z_array[slice, :, 1], ax=ax, t_axis=0, c="red")

timescale = (np.array(range(len(x_array))) / len(x_array) * (stop_iter - start_iter) + start_iter)/1000
timeline = amp.Timeline(timescale, units='K iters', )
animation = amp.Animation([block1, block2], timeline)

fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95, top=0.97)
animation.controls()


#%%

animation.save_gif("plots/DISC_SN_Disc_Layer_norm")

