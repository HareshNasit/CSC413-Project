from typing import List, Callable, Union, Any, TypeVar, Tuple
from enum import Enum
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

class DecoderType(Enum):
    COMBINED = 1
    DOMAIN_X = 2
    DOMAIN_Y = 3