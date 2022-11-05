import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

from utils.gsm import ols_projection, unexplained_variance_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


################################################################################

save_subpath_name = sys.argv[1]
save_subpath = f"save/{save_subpath_name}"
if not os.path.isdir(save_subpath):
    os.mkdir(save_subpath)
    

image_set = torch.tensor(np.load('data/greyscale_cifar/training_images.npy')).float()

projective_fields = torch.tensor(np.load(os.path.join(save_subpath, "projective_fields.npy"))).float()


################################################################################

estimate = ols_projection(projective_fields.reshape(-1, 50), image_set)

fvus = unexplained_variance_loss(image_set, estimate, mean=False).cpu().numpy()

sns.histplot(fvus, bins=500)
plt.savefig(os.path.join(save_subpath, 'fvu_dist.png'))


################################################################################

fig, axes = plt.subplots(5, 5)

for i, ax in enumerate(axes.flatten()):
    
    ax.imshow(projective_fields[:,:,i])

fig.savefig(os.path.join(save_subpath, 'projective_fields.png'))
