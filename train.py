# Adapted from: https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/run.py

import os
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

# from .types_ import *
# from dataset import VAEDataset  # TODO: await Adit
from models import VanillaVAE
from experiment import VAEXperiment



tb_logger =  TensorBoardLogger(save_dir='logs/',
                               name='VanillaVAE')


seed_everything(42, True)

# TODO: decide about these parameters
model = VanillaVAE(
    in_channels=3,
    latent_dim=128,
    hidden_dims=[32, 64, 128, 256, 512]
)

config = {
    'exp_params': {
        'LR': 0.005,
        'weight_decay': 0.0,
        # 'scheduler_gamma': 0.95, # TODO:?
        'kld_weight': 0.00025,
    },
    'data_params': {
        # TODO: decide
        'train_batch_size': 64,
        'val_batch_size':  64,
        'patch_size': 64,
        'num_workers': 4
    }
}

experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)