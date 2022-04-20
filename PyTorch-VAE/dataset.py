import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import random
from models import Domain 


class HorsesAndZebrasDataset(Dataset):
    """
    The Zebra part of the horse2zebra dataset.
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
      
        self.transforms = transform
        self.data_dir = Path(data_path)

        horse_imgs, zebra_imgs = None, None
        horse_path = self.data_dir / ('horse2zebra/testA' if split == 'test' else 'horse2zebra/trainA')
        zebra_path = self.data_dir / ('horse2zebra/testB' if split == 'test' else 'horse2zebra/trainB')
 
        horse_imgs = sorted([f for f in horse_path.iterdir() if f.suffix == '.jpg'])
        zebra_imgs = sorted([f for f in zebra_path.iterdir() if f.suffix == '.jpg'])

        # might not need to shuffle as already done by dataloader
        imgs = horse_imgs + zebra_imgs
        random.shuffle(imgs)
        self.imgs = imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking

class VAEDatasetHorse2Zebra(LightningDataModule):
    """
    PyTorch Lightning data module 
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
    
        train_transforms = transforms.Compose([
          transforms.Resize(self.patch_size),
          transforms.ToTensor()
        ])
        
        val_transforms = transforms.Compose([
          transforms.Resize(self.patch_size),
          transforms.ToTensor()
        ])
        
        self.train_dataset = HorsesAndZebrasDataset(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = HorsesAndZebrasDataset(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

class AppleOrangeDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                 domain: Domain,
                **kwargs):
      
        self.transforms = transform
        self.data_dir = Path(data_path)
        print(self.data_dir)
        domain_X_path = self.data_dir / ('apple2orange/testA' if split == 'test' else 'apple2orange/trainA')
        domain_Y_path = self.data_dir / ('apple2orange/testB' if split == 'test' else 'apple2orange/trainB')

        domain_X_imgs, domain_Y_imgs = [], []
        if domain == Domain.COMBINED:
          domain_X_imgs = sorted([f for f in domain_X_path.iterdir() if f.suffix == '.jpg'])
          domain_Y_imgs = sorted([f for f in domain_Y_path.iterdir() if f.suffix == '.jpg'])
        elif domain == Domain.X:
          domain_X_imgs = sorted([f for f in domain_X_path.iterdir() if f.suffix == '.jpg'])
        elif domain == Domain.Y:
          domain_Y_imgs = sorted([f for f in domain_Y_path.iterdir() if f.suffix == '.jpg'])
 
        self.imgs = domain_X_imgs + domain_Y_imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy data to prevent breaking

class VAEDatasetAppleOrange(LightningDataModule):
    """
    PyTorch Lightning data module 
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        domain: Domain = Domain.COMBINED,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.domain = domain

    def setup(self, stage: Optional[str] = None) -> None:
    
        train_transforms = transforms.Compose([
          transforms.Resize(self.patch_size),
          transforms.ToTensor()
        ])
        
        val_transforms = transforms.Compose([
          transforms.Resize(self.patch_size),
          transforms.ToTensor()
        ])
        
        self.train_dataset = AppleOrangeDataset(
            self.data_dir,
            split='train',
            transform=train_transforms,
            domain=self.domain
        )
        
        self.val_dataset = AppleOrangeDataset(
            self.data_dir,
            split='test',
            transform=val_transforms,
            domain=self.domain
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )