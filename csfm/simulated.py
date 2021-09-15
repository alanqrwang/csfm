import torch
import numpy as np
from csfm.util.train import BaseTrain

class Simulated(BaseTrain):
  """Simulated."""

  def __init__(self, args):
    super(Simulated, self).__init__(args=args)