import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset import VAEDatasetAppleOrange as VAEDataset
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

def create_logger(name):
  logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=name,)
  return logger


# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**{
  **config['model_params'], **config['data_params']
})
experiment = VAEXperiment(model,
                          config['exp_params'])

def create_runner(logger):
  runner = Trainer(
    logger=logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(save_top_k=2, 
                        dirpath =os.path.join(logger.log_dir , "checkpoints"), 
                        monitor= "val_loss",
                        save_last= True),
    ],
    strategy=DDPPlugin(find_unused_parameters=False),
    **config['trainer_params']
  )
  return runner

# 1. Train the full VAE
full_logger = create_logger("full-VAE")
runner = create_runner(full_logger)

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()

Path(f"{full_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{full_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")

runner.fit(experiment, datamodule=data)

# 2. Freeze the encoder
model.freeze_encoder()

# 3. Train two decoders
# 3a. - Domain X
domainX_logger = create_logger("domainX")
runner = create_runner(domainX_logger)

model.set_decoder(DecoderType.DOMAIN_X)
experiment_X = VAEXperiment(model,
                            config['exp_params'])

data_X = VAEDataset(
  **config["data_params"],
  pin_memory=len(config['trainer_params']['gpus']) != 0,
  domain=Domain.X
)
data_X.setup()

Path(f"{domainX_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{domainX_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

runner.fit(experiment_X, datamodule=data_X)

# 3b. - Domain Y
domainY_logger = create_logger('domainY')
runner = create_runner(domainY_logger)

data_Y = VAEDataset(
  **config["data_params"],
  pin_memory=len(config['trainer_params']['gpus']) != 0,
  domain=Domain.Y
)
data_Y.setup()


Path(f"{domainY_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{domainY_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
# Use decoder Y
model.set_decoder(DecoderType.DOMAIN_Y)
experiment_X_to_Y = VAEXperiment(
  model,
  config['exp_params']
)
runner.fit(experiment_X_to_Y, datamodule=data_Y)


