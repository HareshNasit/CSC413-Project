import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from pytorch_msssim import ms_ssim


class Encoder(nn.Module):
  def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List,
        patch_size_hidden: int
    ):
      super().__init__()

      self.patch_size_hidden = patch_size_hidden

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

      self.model = nn.Sequential(*modules)
      self.fc_mu = nn.Linear(hidden_dims[-1] * (self.patch_size_hidden**2), latent_dim)
      self.fc_var = nn.Linear(hidden_dims[-1] * (self.patch_size_hidden**2), latent_dim)

  def forward(self, input: Tensor) -> List[Tensor]:
      result = self.model(input)
      result = torch.flatten(result, start_dim=1)

      # Split the result into mu and var components
      # of the latent Gaussian distribution
      mu = self.fc_mu(result)
      log_var = self.fc_var(result)

      return [mu, log_var]

  def freeze_weights(self):
    for param in self.model.parameters():
      param.requires_grad = False


class Decoder(nn.Module):
  def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        hidden_dims: List,
        patch_size_hidden: int
    ):
      super().__init__()

      self.patch_size_hidden = patch_size_hidden
      self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * (self.patch_size_hidden ** 2))

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

      self.final_layer = nn.Sequential(
                          nn.ConvTranspose2d(hidden_dims[-1],
                                              hidden_dims[-1],
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1),
                          nn.BatchNorm2d(hidden_dims[-1]),
                          nn.LeakyReLU(),
                          nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                                    kernel_size= 3, padding= 1),
                          nn.Tanh())
  
  def forward(self, z: Tensor) -> Tensor:
      result = self.decoder_input(z)
      result = result.view(-1, 512, self.patch_size_hidden, self.patch_size_hidden)
      
      result = self.decoder(result)
      result = self.final_layer(result)
      return result


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_channels = in_channels

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Compute hidden patch size
        conv_layer_count = len(hidden_dims)
        self.patch_size_hidden = kwargs['patch_size'] // (2**conv_layer_count)

        # Build Encoder
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims, self.patch_size_hidden)

        # Build Decoder
        decoder_params = [in_channels, latent_dim, list(reversed(hidden_dims)), self.patch_size_hidden]
        self.decoders = {
            DecoderType.COMBINED: Decoder(*decoder_params),
            DecoderType.DOMAIN_X: Decoder(*decoder_params),
            DecoderType.DOMAIN_Y: Decoder(*decoder_params)
        }
        self.decoder = self.decoders[DecoderType.COMBINED]

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

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
    
    def set_decoder(self, decoder_type: DecoderType):
        self.decoder = self.decoders[decoder_type]
    
    def freeze_encoder(self):
        self.encoder.freeze_weights()

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

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
        loss_l2 = F.mse_loss(recons, input)
        loss_ms_ssim = 1 - ms_ssim(recons, input, data_range=1, size_average=True)
        recons_loss = loss_l2 + loss_ms_ssim

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
