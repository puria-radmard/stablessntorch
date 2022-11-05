import numpy as np
import torch
import json
import sys
import os
from models.GSM import GSM
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


################################################################################

save_subpath_name = sys.argv[1]
save_subpath = f"save/{save_subpath_name}"
if not os.path.isdir(save_subpath):
    os.mkdir(save_subpath)
    

################################################################################
############  TASK GENERATION  #################################################
################################################################################

## Initialise
square_size = 32
n_filters = 50
image_set = torch.tensor(np.load('data/greyscale_cifar/training_images.npy'))
gsm_model = GSM(n_filters=n_filters, square_size=square_size).to(device)

if sys.argv[2] == 'train':
    ## Train GSM projective fields and save
    A_loss_history = gsm_model.train_projective_fields(image_set=image_set)
    with open(os.path.join(save_subpath, "A_loss_history.json"), "w") as f:
        json.dump(A_loss_history, f)
    np.save(os.path.join(save_subpath, "projective_fields.npy"), gsm_model.projective_fields.cpu().numpy())

    ## Train GSM bayesian parameters and save
    gsm_model.initialise_bayesian_params(image_set=image_set)
    gsm_model.train_bayesian_parameters(image_set=image_set)

    torch.save(gsm_model.state_dict(), os.path.join(save_subpath, "gsm_state_dict.mdl"))

else:
    gsm_model.projective_fields = torch.tensor(np.load(os.path.join(save_subpath, "projective_fields.npy")))
    gsm_model.load_state_dict(torch.load(os.path.join(save_subpath, "gsm_state_dict.mdl")))


## Infer target distributions for SSN and save
(
    posterior_distribution_means,
    posterior_distribution_covs,
) = gsm_model.infer_latent_distribution(image_set)
np.save(os.path.join(save_subpath, "gsm_posterior_distribution_means.npy"), posterior_distribution_means.numpy())
np.save(os.path.join(save_subpath, "gsm_posterior_distribution_covs.npy"), posterior_distribution_covs.numpy())

## Infer inputs for SSN and save
input_features = gsm_model.generate_ssn_inputs(image_set, make_positive=True).numpy()
np.save(os.path.join(save_subpath, "ssn_inputs.npy"), input_features)
