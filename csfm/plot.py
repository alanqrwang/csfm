import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plotloss(model_dirs, labels=None, ylim=None, xlim=None):
  if not isinstance(model_dirs, (tuple, list)):
    model_dirs = [model_dirs]
  if not isinstance(labels, (tuple, list)):
    labels = [labels]
  
  if labels[0] is None:
    labels = ['Line %d' % n for n in range(len(model_dirs))]
  assert len(labels) == len(model_dirs), 'labels do not match model paths'

  fig, axes = plt.subplots(1, 1)
  fig.set_size_inches(18.5, 10.5)
  for i, model_dir in enumerate(model_dirs):
    pkl_path = os.path.join(model_dir, 'checkpoints', 'losses.pkl')
    with (open(pkl_path, "rb")) as openfile:
      loss_dict = pickle.load(openfile)
    loss = loss_dict['loss']
    val_loss = loss_dict['val_loss']
 
    color = next(axes._get_lines.prop_cycler)['color']
    xvalues = np.arange(1, len(loss)+1) 
    axes.plot(xvalues, loss, color=color, label=labels[i])
    axes.plot(xvalues, val_loss, color=color, linestyle='dashed')
 
    if ylim is not None:
        axes.set_ylim(ylim)
    if xlim is not None:
        axes.set_xlim(xlim)
    axes.grid()
    axes.legend()