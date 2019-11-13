## This is a project for experiments with GAN's
* ##### Pytorch implementation
* ##### Gan's are trained on toy datasets: gaussian grid, gaussian circle, gaussian spiral
* ##### There are spectral normalization and gradient penalty options with different loss configurations 
* ##### Implemented WGAN-div, LSGAN, original GAN and relativistic GAN
* ##### Also there is code for visualizing gan training

### Explore [src/toygan.py](/src/toygan.py) for more details

#### Spectral normalization in discriminator with gradient penalty.
![](/gifs/DISC_SN_looped.gif)

#### Same with layer normalization in generator
![](/gifs/DISC_SN_Gen_Layer_norm_looped.gif)
