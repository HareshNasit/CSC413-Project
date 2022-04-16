import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from dataset import VAEDataset, VAEDataset2, VAEDataset3
from dataset import VAEDatasetMNIST as VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**{
  **config['model_params'], **config['data_params']
})
experiment = VAEXperiment(model,
                          config['exp_params'])

# data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
# TODO: we added
# data = VAEDataset2(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
# data = VAEDataset3(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

create_runner = lambda: Trainer(
  logger=tb_logger,
  callbacks=[
      LearningRateMonitor(),
      ModelCheckpoint(save_top_k=2, 
                      dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                      monitor= "val_loss",
                      save_last= True),
  ],
  strategy=DDPPlugin(find_unused_parameters=False),
  **config['trainer_params']
)

# 1. Train the full VAE
runner = create_runner()

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")

runner.fit(experiment, datamodule=data)  # TODO: uncomment

# 2. Freeze the encoder
model.freeze_encoder()

# 3. Train two decoders
# 3a. - Domain X
runner = create_runner()

model.set_decoder(DecoderType.DOMAIN_X)
experiment_X = VAEXperiment(model,
                            config['exp_params'])

data_X = VAEDataset(
  **config["data_params"],
  pin_memory=len(config['trainer_params']['gpus']) != 0,
  domain=Domain.X
)
data_X.setup()

runner.fit(experiment_X, datamodule=data_X)

# 3b. - Domain Y
# TODO: do same for Y

# 4. Image to image translation (X -> Y) and (Y -> X)
# Sample from X
runner = create_runner()

data_X = VAEDataset(
  **config["data_params"],
  pin_memory=len(config['trainer_params']['gpus']) != 0,
  domain=Domain.X
)
data_X.setup()

# Use decoder Y
model.set_decoder(DecoderType.DOMAIN_Y)
experiment_X_to_Y = VAEXperiment(
  model,
  config['exp_params']
)
runner.fit(experiment_X_to_Y, datamodule=data_X)