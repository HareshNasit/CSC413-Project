# Retrieved from: https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/

from typing import List, Callable, Union, Any, TypeVar, Tuple
from enum import Enum
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

class DecoderType(Enum):
    COMBINED = 1
    DOMAIN_X = 2
    DOMAIN_Y = 3