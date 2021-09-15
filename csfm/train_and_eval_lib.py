"""Utilities to run model training and evaluation."""

from csfm.raw import Raw
from csfm.simulated import Simulated

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'raw':
    trainer = Raw(args)
  elif args.method.lower() == 'simulated':
    trainer = Simulated(args)
  else:
    raise NotImplementedError
  return trainer