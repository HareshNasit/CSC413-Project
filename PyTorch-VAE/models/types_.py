from typing import List, Callable, Union, Any, TypeVar, Tuple
from enum import Enum
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

class DecoderType(Enum):
  # TODO: use Domain?
  COMBINED = 1
  DOMAIN_X = 2
  DOMAIN_Y = 3

class Domain(Enum):
  COMBINED = 1
  X = 2
  Y = 3