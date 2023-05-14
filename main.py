import matplotlib.pyplot as plt

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

def show_tensor_images(image_tensor,num_images=25,size =(1,28,28)):
    '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

class Generator(nn.Module):
    '''
        Generator Class
        Values:
            z_dim: the dimension of the noise vector, a scalar
            im_chan: the number of channels in the images, fitted for the dataset used, a scalar
                  (MNIST is black-and-white, so 1 channel is your default)
            hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self,z_dim,im_chan=1,hidden_dim = 64):
        super(Generator,self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim,4*hidden_dim),
            self.make_gen_block(4*hidden_dim,2*hidden_dim),
            self.make_gen_block(2*hidden_dim,hidden_dim),
            self.make_gen_block(hidden_dim,im_chan,final_layer = True)

        )


