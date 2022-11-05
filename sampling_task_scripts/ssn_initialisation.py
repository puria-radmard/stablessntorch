import numpy as np
import torch
from models.SSN import SameNumEISSN
from utils.dynamics_neutral_growth import select_top_params


## Reload

posterior_distribution_means = torch.tensor(np.load("save/gsm_posterior_distribution_means.npy"))
posterior_distribution_covs = torch.tensor(np.load("save/gsm_posterior_distribution_covs.npy"))

input_features = torch.tensor(np.load("save/ssn_inputs.npy"))

################################################################################
############  SSN INITIALISATION  ##############################################
################################################################################

# Constants and model class
dt = 0.0002
tau_eta = 0.02
simulation_T = 0.5
num_initialisation_patterns = 100
num_initialisation_iterations=10
repeats_per_initialisation_iteration=20

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
)

# Targets for initialise training - first 100 images with only first dimenion taken out
inith = input_features[:num_initialisation_patterns, :1]
initmean = posterior_distribution_means[:num_initialisation_patterns, :1]
initcov = posterior_distribution_covs[:num_initialisation_patterns, :1, :1]


# Train initial weight matrix
# TODO: understand and num_iterations and repeats_per_iteration better
# All are [num_iterations, repeats_per_iteration, ...]
init_costs, init_ws, init_thetas = ssn.initial_weight_training(
    init_input = inith,
    target_cov = initcov,
    target_mean = initmean,
    num_iterations = num_initialisation_iterations,
    repeats_per_iteration = repeats_per_initialisation_iteration,
    num_patterns = num_initialisation_patterns,
    dt = dt,
)
np.save("save/parameters/init_costs.npy", init_costs.detach().numpy())
np.save("save/parameters/init_ws.npy", init_ws.detach().numpy())
np.save("save/parameters/init_thetas.npy", init_thetas.detach().numpy())

top_costs, top_ws, top_thetas, ranking = select_top_params(init_costs, init_ws, init_thetas, num_initialisation_iterations)

np.save('./parameters/top_init_ws.npy',top_ws[ranking[:10]].detach().numpy())
np.save('./parameters/top_init_thetas.npy',top_thetas[ranking[:10]].detach().numpy())



################################################################################
############  SSN FINAL SAVE  ##################################################
################################################################################

ssn.load_N_matrix(0.9*torch.identity(2) + 0.1*torch.ones([2,2]))
ssn.load_thetas(top_thetas[0].detach())
ssn.load_W(top_ws.detach())

# Save so we can skip directly to here later
torch.save(ssn.state_dict(), "save/network_growth/simexpand_E1.mdl")
