from torch import nn
import pytorch_lightning as pl
from models.vanilla_vae import VanillaVAE
from experiment import VAEXperiment
import torch
chk_path = "/work/CSC413-Project/logs/VanillaVAE/version_0"

full_chk_path = "/content/CSC413-Project/PyTorch-VAE/logs/full-VAE/version_1/checkpoints/full_VAE_last.ckpt"
apple_chk_path = "/content/CSC413-Project/PyTorch-VAE/logs/domainX/version_1/checkpoints/apple_VAE_last.ckpt"

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

#apple_model.load_state_dict(torch.load(apple_chk_path))
full_vae_checkpoint = torch.load(full_chk_path)
apple_vae_checkpoint = torch.load(apple_chk_path)
print("\n\n\n\n\n checkpoint state_dict")
#print(full_vae_checkpoint["state_dict"])
full_model.load_state_dict(full_vae_checkpoint["state_dict"])
apple_model.load_state_dict(apple_vae_checkpoint["state_dict"])
print(full_model.state_dict)
# vae_test = VanillaVAE(
#     # in_channels=1,
#     in_channels=3,  # TODO: put back
#     # latent_dim=128, 
#     latent_dim=128,
#     hidden_dims=[32, 64, 128, 256, 512],
#     patch_size=256
#     # hidden_dims=[32, 64, 128]
# )
# print(vae_test.state_dict)

# full_model = VAEXperiment.load_from_checkpoint(checkpoint_path=full_chk_path)

# apple_model = apple_model.load_from_checkpoint(apple_chk_path, exp_params)






