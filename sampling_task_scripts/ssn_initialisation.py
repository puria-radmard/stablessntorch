import numpy as np
import torch
import sys, os
from models.SSN import SameNumEISSN
from utils.dynamics_neutral_growth import select_top_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'


################################################################################

save_subpath_name = sys.argv[1]
save_subpath = f"save/{save_subpath_name}"
if not os.path.isdir(save_subpath):
    os.mkdir(save_subpath)
if not os.path.isdir(os.path.join(save_subpath, "network_growth")):
    os.mkdir(os.path.join(save_subpath, "network_growth"))

################################################################################


## Reload

posterior_distribution_means = torch.tensor(np.load(os.path.join(save_subpath, "gsm_posterior_distribution_means.npy")))
posterior_distribution_covs = torch.tensor(np.load(os.path.join(save_subpath, "gsm_posterior_distribution_covs.npy")))

input_features = torch.tensor(np.load(os.path.join(save_subpath, "ssn_inputs.npy")))

################################################################################
############  SSN INITIALISATION  ##############################################
################################################################################

# Constants and model class
dt = 0.0002
tau_eta = 0.02
simulation_T = 0.5
num_initialisation_patterns = 100
num_initialisation_iterations = 10
repeats_per_initialisation_iteration = 20

# Initiliase model class instance
ssn = SameNumEISSN(
    lambda_mean=0.01,
    lambda_var=0.02,
    lambda_cov=0.01,
    tau_e=0.020,
    tau_i=0.010,
    k=0.3,
    gamma=2.0,
    eps1=(1.0 - dt / tau_eta),
    eps2=np.sqrt(2.0 * dt / tau_eta),
    num_initial_e_neurons=1,
).to(device)

# Targets for initialise training - first 100 images with only first dimenion taken out
inith = input_features[:num_initialisation_patterns, :1].to(device)
initmean = posterior_distribution_means[:num_initialisation_patterns, :1].to(device)
initcov = posterior_distribution_covs[:num_initialisation_patterns, :1, :1].to(device)


# Train initial weight matrix
# All are [num_iterations, repeats_per_iteration, ...]
init_costs, init_ws, init_thetas = ssn.initial_weight_training(
    init_input = inith.float(),
    target_cov = initcov.float(),
    target_mean = initmean.float(),
    num_iterations = num_initialisation_iterations,
    repeats_per_iteration = repeats_per_initialisation_iteration,
    num_patterns = num_initialisation_patterns,
    dt = dt,
)
np.save(os.path.join(save_subpath, "init_costs.npy"), init_costs.detach().cpu().numpy())
np.save(os.path.join(save_subpath, "init_ws.npy"), init_ws.detach().cpu().numpy())
np.save(os.path.join(save_subpath, "init_thetas.npy"), init_thetas.detach().cpu().numpy())

top_costs, top_ws, top_thetas, ranking = select_top_params(init_costs, init_ws, init_thetas, num_initialisation_iterations)

np.save(os.path.join(save_subpath, 'top_init_ws.npy'), top_ws[ranking[:10]].detach().cpu().numpy())
np.save(os.path.join(save_subpath, 'top_init_thetas.npy'), top_thetas[ranking[:10]].detach().cpu().numpy())



################################################################################
############  SSN FINAL SAVE  ##################################################
################################################################################

ssn.load_N_matrix(0.9*torch.eye(2) + 0.1*torch.ones([2,2]))
ssn.load_thetas(top_thetas[ranking[0]].detach().cpu())
ssn.load_W(top_ws[ranking[0]].detach().cpu())

# Save so we can skip directly to here later
torch.save(ssn.state_dict(), os.path.join(save_subpath, "network_growth/simexpand_E1.mdl"))

import pdb; pdb.set_trace()
