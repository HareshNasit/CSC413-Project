from typing import List, Callable, Union, Any, TypeVar, Tuple
from enum import Enum

Tensor = TypeVar('torch.tensor')

class DecoderType(Enum):
  COMBINED = 1
  DOMAIN_X = 2
  DOMAIN_Y = 3

class Domain(Enum):
  COMBINED = 1
  X = 2
  Y = 3