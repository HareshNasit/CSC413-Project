from torch import nn
from models import *
import pytorch_lightning as pl
from models.vanilla_vae import VanillaVAE
from dataset import VAEDatasetAppleOrange as VAEDataset
from experiment import VAEXperiment
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import torch
import torchvision

chk_path = "/work/CSC413-Project/logs/VanillaVAE/version_0"

full_chk_path = "/content/full_VAE_last.ckpt"
apple_chk_path = "/content/apple_VAE_last.ckpt"

exp_params = {
  "LR": 0.0005,
  "weight_decay": 0.0,
  "scheduler_gamma": 0.95,
  "kld_weight": 0.00025,
  "manual_seed": 1265
}

full_model = VAEXperiment(VanillaVAE(
    # in_channels=1,
    in_channels=3,  # TODO: put back
    # latent_dim=128, 
    latent_dim=128,
    hidden_dims=[32, 64, 128, 256, 512],
    patch_size=256
    # hidden_dims=[32, 64, 128]
), exp_params)

apple_model = VAEXperiment(VanillaVAE(
    # in_channels=1,
    in_channels=3,  # TODO: put back
    # latent_dim=128, 
    latent_dim=128,
    hidden_dims=[32, 64, 128, 256, 512],
    patch_size=256
    # hidden_dims=[32, 64, 128]
), exp_params)

# load checkpoints
full_vae_checkpoint = torch.load(full_chk_path)
apple_vae_checkpoint = torch.load(apple_chk_path)
# use pretrained weights
full_model.load_state_dict(full_vae_checkpoint["state_dict"])
apple_model.load_state_dict(apple_vae_checkpoint["state_dict"])



# current test take orange image through full model ENCODER.
data_Y = VAEDataset(
  data_path='Data/',
  domain=Domain.Y
)
data_Y.setup()
data_loader = data_Y.test_dataloader()
test_orange_batches = []
for test_images, test_labels in data_loader:  
  test_orange_batches.append(test_images)
print(test_orange_batches[0].shape)

output_images = apple_model.forward(test_orange_batches[0])
# print(len(input_hidden_states))
# print((input_hidden_states[0].shape))
# print((input_hidden_states[1].shape))
# take the hidden state and put it through apple VAE DECODER.

print(len(output_images[0]))
# visualize result.
# tensor_image = output_images[0][0]

# reshape to channel first:
# So we need to reshape it to (H, W, C):
# tensor_image = tensor_image.view(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
# print(type(tensor_image), tensor_image.shape)
# tensor_image = tensor_image.detach().numpy()
# plt.imshow(tensor_image)
# plt.show()
# download 10 of the original images
for i in range(10):
  save_image(test_orange_batches[0][i], 'orange' + str(i) + '.png')
for i in range(10):
  save_image(output_images[0][i], 'apple' + str(i) + '.png')





