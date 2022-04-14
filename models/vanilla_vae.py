# Adapted from: https://github.com/AntixK/PyTorch-VAE/

import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List
    ):
        modules = []  
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)  # TODO: change 4 for whatever the flattened latent dimension is
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def freeze_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: List
        latent_dim: int,
    ):
        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 hidden_dims: List,
                 latent_dim: int,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.encoder = Encoder(
            in_channels,
            hidden_dims,
            latent_dim
        )
        decoder_main = Decoder(
            in_channels,
            reversed(hidden_dims),
            latent_dim
        )
        decoder_style1 = Decoder(
            in_channels,
            reversed(hidden_dims),
            latent_dim
        )
        decoder_style2 = Decoder(
            in_channels,
            reversed(hidden_dims),
            latent_dim
        )

        self.decoders = {
            0: decoder_main,
            1: decoder_style1,
            2: decoder_style2
        }

        self.latent_dim = latent_dim

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        # TODO: use something else?
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],  # TODO: probably hidden_dims[0]
                                               hidden_dims[-1],  # TODO: probably hidden_dims[0]
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def freeze_encoder(self):
        self.encoder.freeze_weights()

    def forward(self, input_: Tensor, **kwargs, decoder: Decoder, decoder_num: int) -> List[Tensor]:

        # Run encoder
        mu, log_var = self.encoder(input_)
        z = self.reparameterize(mu, log_var)
        
        # Run decoder
        decoder = self.decoders[decoder_num]
        
        return  [decoder(z), input_, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

def train():
    model = VanillaVAE(
        # ...
    )
    device = # ...
    
