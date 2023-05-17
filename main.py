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
            self.make_gen_block(4*hidden_dim,2*hidden_dim,kernel_size = 4,stride = 1),
            self.make_gen_block(2*hidden_dim,hidden_dim),
            self.make_gen_block(hidden_dim,im_chan,kernel_size=4,final_layer = True)

        )

    def make_gen_block(self,input_channels,output_channels,kernel_size =3,stride=2,final_layer = False):
        '''

                This function defines one layer in the sequential generator block. It includes a
                transposed convolutional layer, batch normalization and ReLU activation.

                Inputs:
                input_channels: The number of channels of the input feature
                output_channels: The number of channels the output feature should have
                kernel_size: the size of the convolutional filters (kernel_size,kernel_size)
                stride: the covolutional stride
                final_layer: a boolean value, True for the final layer of the network,otherwise false

                Output:
                returns a single deconvolution layer for the network
        '''

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace= True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.Tanh()
            )

    def forward(self,noise):
        '''

        Defines a single forward pass of the generator. Returns generated image from the input noise

        Inputs:
        noise: a noise vector with dimensions (n_samples,z_dim)

        Output:
        returns fake images generated from the generated from the generator from the noise
        '''

        return self.gen(noise.view(len(noise),self.z_dim,1,1))

def generate_noise(n_samples,z_dim):
    '''
    returns a random vector of dimension (n_samples,z_dim)

    Inputs:
    n_samples: number of images to be generated
    z_dim: a scaler, length of the noise vector

    Output:
    returns the noise vector according to the input dimensions
    '''

    return torch.randn(n_samples,z_dim)

class Discriminator(nn.Module):
    def __init__(self,im_chan=1,hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan,hidden_dim),
            self.make_disc_block(hidden_dim,hidden_dim*2),
            self.make_disc_block(hidden_dim*2,1,final_layer = True)
        )
    def make_disc_block(self,im_chan,hidden_dim=64,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels=im_chan,out_channels=hidden_dim,kernel_size=4,stride=2),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(0.2,inplace = True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=im_chan,out_channels=1,kernel_size=4,stride=2)
            )

    def forward(self,image):
        x = self.disc(image)
        return x.view(len(x),-1)

criterion = nn.BCEWithLogitsLoss()
z_dim = 64
batch_size = 128
display_step = 500
lr = 0.0002

beta_1 = 0.5
beta_2 = 0.999
device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

dataloader = DataLoader(
    MNIST('.',download = True,transform=transform),
    batch_size = batch_size,shuffle=True
)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(),lr=lr,betas=(beta_1,beta_2))
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(),lr = lr,betas=(beta_1,beta_2))


def initiate_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(initiate_weights)
disc = disc.apply(initiate_weights)

n_epochs = 50
cur_step = 0
mean_gen_loss = 0
mean_disc_loss = 0

for epoch in range(n_epochs):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)
        disc_opt.zero_grad()
        fake_noise = generate_noise(cur_batch_size,z_dim)
        fake_image = gen(fake_noise)
        fake_pred = disc(fake_image.detach())
        disc_fake_loss = criterion(fake_pred,torch.zeros_like(fake_pred))
        real_pred = disc(real)
        disc_real_loss = criterion(real_pred,torch.ones_like(real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        mean_disc_loss += disc_loss.item() / display_step
        disc_loss.backward(retain_graph = True)
        disc_opt.step()

        gen_opt.zero_grad()
        fake_noise_2 = generate_noise(cur_batch_size,z_dim)
        fake_image_2 = gen(fake_noise_2)
        fake_pred_2 = disc(fake_image_2)
        gen_loss = criterion(fake_pred_2,torch.ones_like(fake_pred_2))

        mean_gen_loss += gen_loss.item() /display_step
        gen_loss.backward()
        gen_opt.step()

        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_gen_loss}, discriminator loss: {mean_disc_loss}")
            show_tensor_images(fake_image)
            show_tensor_images(real)
            mean_gen_loss = 0
            mean_disc_loss = 0
        cur_step += 1

