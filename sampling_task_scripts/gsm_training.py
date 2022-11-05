import numpy as np
import torch
import json
import sys
from models.GSM import GSM


################################################################################
############  TASK GENERATION  #################################################
################################################################################

## Initialise
square_size = 32
n_filters = 50
image_set = torch.tensor(np.load('/Users/puriaradmard/Documents/GitHub/stableSSNTorch/data/greyscale_cifar/training_images.npy'))
image_set = image_set[:2121]
gsm_model = GSM(n_filters=n_filters, square_size=square_size)

if sys.argv[1] == 'retrain':
    ## Train GSM projective fields and save
    A_loss_history = gsm_model.train_projective_fields(image_set=image_set)
    with open("save/A_loss_history.json", "w") as f:
        json.dump(A_loss_history, f)
    np.save("save/projective_fields.npy", gsm_model.projective_fields.numpy())

    ## Train GSM bayesian parameters and save
    gsm_model.initialise_bayesian_params(image_set=image_set)
    gsm_model.train_bayesian_parameters(image_set=image_set)

    torch.save(gsm_model.state_dict(), "save/gsm_state_dict.mdl")

else:
    gsm_model.load_state_dict(torch.load("save/gsm_state_dict.mdl"))
    gsm_model.projective_fields = torch.tensor(np.load("save/projective_fields.npy"))


## Infer target distributions for SSN and save
(
    posterior_distribution_means,
    posterior_distribution_covs,
) = gsm_model.infer_latent_distribution(image_set)
np.save("save/gsm_posterior_distribution_means.npy", posterior_distribution_means.numpy())
np.save("save/gsm_posterior_distribution_covs.npy", posterior_distribution_covs.numpy())
# TODO: make target vars here too


## Infer inputs for SSN and save
input_features = gsm_model.generate_ssn_inputs(image_set, make_positive=True).numpy()
np.save("save/ssn_inputs.npy", input_features)
