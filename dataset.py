from PIL import Image
import os
import torch
import random
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HorsesDataset(Dataset):
    """
    The Horse part of the horse2zebra dataset.
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        if split == 'test':
          self.data_dir = Path(data_path) / 'horse2zebra/testA'
        else:
          self.data_dir = Path(data_path) / 'horse2zebra/trainA'      
        self.transforms = transform

        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking

class ZebrasDataset(Dataset):
    """
    The Zebra part of the horse2zebra dataset.
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        if split == 'test':
          self.data_dir = Path(data_path) / 'horse2zebra/testB'
        else:
          self.data_dir = Path(data_path) / 'horse2zebra/trainB'      
        self.transforms = transform

        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking

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

class VAEDataset(LightningDataModule):
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
    
        train_transforms = transforms.Compose([transforms.ToTensor()])
        
        val_transforms = transforms.Compose([transforms.ToTensor()])
        
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

