import numpy as np
import torch
import os
import sys

from tqdm import tqdm
from models.SSN import SameNumEISSN
from utils.dynamics_neutral_growth import dynamics_neutral_mitosis, select_top_params


################################################################################

save_subpath_name = sys.argv[2]
save_subpath = f"save/{save_subpath_name}"
if not os.path.isdir(save_subpath):
    os.mkdir(save_subpath)

################################################################################


## Reload
posterior_distribution_means = torch.tensor(np.load(os.path.join(save_subpath, "gsm_posterior_distribution_means.npy")))
posterior_distribution_covs = torch.tensor(np.load(os.path.join(save_subpath, "gsm_posterior_distribution_covs.npy")))

input_features = torch.tensor(np.load(os.path.join(save_subpath, "ssn_inputs.npy")))


# Number of neurons we start with, i.e. we train with starting_size E and I neurons then grow one more
# i.e. W shape is [2*starting_size, 2*starting_size]
starting_size = int(sys.argv[1])

################################################################################
############  SSN RESUMING  ####################################################
################################################################################

# Constants and model class
dt = 0.0002
tau_eta = 0.02


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
    num_initial_e_neurons=starting_size,
)

ssn.load_state_dict(torch.load(os.path.join(save_subpath, f"network_growth/simexpand_E{starting_size}.mdl")))

################################################################################
############  SSN TRAINING  ####################################################
################################################################################


if starting_size<5:     
    multiplier = 2
elif starting_size<10:  
    multiplier = 10
elif starting_size<20:  
    multiplier = 5
else:                   
    multiplier = 4

simulation_T = 0.5              # Ignore this?
num_patterns_per_iter = 100
num_iters_total = 50
total_patterns = posterior_distribution_means.shape[0]
num_burn_in_step = 2000
num_steps = 500       # actual

num_trials = 100    # subbatchsize


optimiser = torch.optim.Adam(ssn.parameters(), lr=0.01 if starting_size < 4 else 0.0001)

for i in tqdm(range(num_iters_total)):

    optimiser.zero_grad()
    
    chosen_patterns = (torch.rand(num_patterns_per_iter) * (total_patterns-1)).int()

    batch_posterior_distribution_means = torch.index_select(posterior_distribution_means, 0, chosen_patterns)[:,:starting_size]
    batch_posterior_distribution_covs  = torch.index_select(posterior_distribution_covs, 0, chosen_patterns)[:,:starting_size, :starting_size]
    batch_input_features               = torch.index_select(input_features, 0, chosen_patterns)[:,:starting_size]

    iteration_cost = torch.tensor(0.)

    for j in range(int(20/multiplier)):

        target_means_j = batch_posterior_distribution_means[j*5*multiplier:(j+1)*5*multiplier]
        target_covs_j = batch_posterior_distribution_covs[j*5*multiplier:(j+1)*5*multiplier]
        input_j = batch_input_features[j*5*multiplier:(j+1)*5*multiplier]

        u_history = ssn.run_dynamics(
            num_trials   = num_trials,
            num_patterns = 5*multiplier,
            num_steps    = num_burn_in_step + num_steps,
            dt           = dt,
            h            = batch_input_features
        )

        uall = torch.stack(u_history[num_burn_in_step:], 0)

        cost = ssn.full_loss_function(uall, target_means_j, target_covs_j)
        
        iteration_cost += multiplier * cost / 20

    iteration_cost.backward()

    optimiser.step()


################################################################################
############  SSN GROWTH  ######################################################
################################################################################

if starting_size < 50:

    new_W = dynamics_neutral_mitosis(ssn.W.data, starting_size-1)
    new_W = dynamics_neutral_mitosis(     new_W, starting_size*2)
    ssn.load_W(new_W)

    new_N = dynamics_neutral_mitosis(ssn.N_matrix.data, starting_size-1)
    new_N = dynamics_neutral_mitosis(            new_N, starting_size*2)
    ssn.load_N_matrix(new_N)


    torch.save(
        ssn.state_dict(), os.path.join(save_subpath, f"network_growth/simexpand_E{starting_size+1}.mdl")
    )
