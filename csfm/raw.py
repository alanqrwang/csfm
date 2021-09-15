import torch
import numpy as np
from csfm.util.train import BaseTrain

class Raw(BaseTrain):
  """Raw."""

  def __init__(self, args):
    super(Raw, self).__init__(args=args)