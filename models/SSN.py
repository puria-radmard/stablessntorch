import torch
from torch import nn
from torch import Tensor as T
from torch.nn.parameter import Parameter, UninitializedParameter

from typing import Dict

from tqdm import tqdm

from utils.ssn import dales_mask, dynamics_step, full_loss_function, inhibitory_cost, input_nonlinearity, sampling_cost_from_statistics


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SameNumEISSN(nn.Module):
    """
    Same number of E and I at all times. Allows network growth, default starting with 1 E and 1 I neuron.
    This is determined by W_init
    dynamics_kwargs contains tau_e, tau_i, k, gamma, eps1, eps2 from equation (1)

    # TODO:
    #   - Understand and add multiplier parameter
    #   - Clip u at each step
    #   - Smooth out eta noise
    """

    def __init__(self, lambda_mean: float, lambda_var: float, lambda_cov: float, num_initial_e_neurons: int = 1,  **dynamics_kwargs) -> None:

        super(SameNumEISSN, self).__init__()

        self.lambda_mean = lambda_mean
        self.lambda_var  = lambda_var
        self.lambda_cov  = lambda_cov

        self.num_initial_e_neurons = num_initial_e_neurons

        self.dynamics_kwargs = dynamics_kwargs

        # Will be directly overwritten later
        self.W = Parameter(torch.zeros([num_initial_e_neurons * 2, num_initial_e_neurons * 2]))
        self.thetas = Parameter(torch.zeros(3))
        self.N_matrix = Parameter(torch.zeros([num_initial_e_neurons * 2, num_initial_e_neurons * 2]))

    def load_W(self, new_W: T):
        self.W = Parameter(new_W)

    def load_N_matrix(self, new_N: T):
        self.N_matrix = Parameter(new_N)

    def load_thetas(self, new_thetas: T):
        self.thetas = Parameter(new_thetas)

    def get_num_neurons(self):
        return self.W.shape[0]

    def mask_weight(self):
        """
        Assuming same number of E and I neurons
        """
        return dales_mask(self.W, self.get_num_neurons() // 2)

    def calculate_cost(self, u_mean: T, target_mean: T, u_cov: T, target_cov: T):
        sampling_cost_from_statistics(
            u_mean, target_mean, u_cov, target_cov, 
            lambda_cov=self.lambda_cov, lambda_var=self.lambda_var, lambda_mean=self.lambda_mean
        )

    def full_loss_function(self, u_hists: T, target_mean: T, target_cov: T):
        return full_loss_function(
            u_hists, 
            target_mean, 
            target_cov, 
            first_n = self.get_num_neurons() // 2,
            lambda_cov=self.lambda_cov, 
            lambda_var=self.lambda_var, 
            lambda_mean=self.lambda_mean
        )

    def update_inh(
        self, 
        init_input: T, 
        target_mean: T,
        target_cov: T,
        num_trials: int, 
        num_steps: int, 
        num_patterns: int, 
        num_burn_in_step: int, 
        dt: float
    ):
        u_history = self.run_dynamics(
            num_trials=num_trials,
            num_patterns=num_patterns,
            num_steps=num_steps, 
            dt=dt, 
            h=init_input,
            num_burn_in_steps=num_burn_in_step
        )

        # [num_steps, num_patterns, num_neurons, num_trials]
        uall = torch.stack(u_history, 0)
        cost = self.full_loss_function(uall, target_mean, target_cov)

        num_i_neurons = self.get_num_neurons() // 2

        i_u_hists = uall[:,:,num_i_neurons:,:]      # [num_steps, num_patters, num I neurons, num_trials]
        newinhmean = i_u_hists.mean(-1).mean(0)       # [num patterns, num I neurons]
        newinhvar = i_u_hists.var(-1).mean(0)         # [num patterns, num I neurons]

        return newinhmean, newinhvar, cost


    def train_init_w_with_inh(self, input_h: T, target_mean: T, target_cov: T, inhmean: T, inhvar: T):

        optimiser = torch.optim.Adam(self.parameters(), lr=0.01)   # Hardcoded!

        target_var = torch.diagonal(target_cov, 0, 2) 

        targetvar2 = torch.concat([target_var, inhvar], axis=1).unsqueeze(-1)
        targetmean2 = torch.concat([target_mean, inhmean], axis=1).unsqueeze(-1)

        for _ in range(70): # Hardcoded!

            optimiser.zero_grad()

            W = self.mask_weight()
            f = input_nonlinearity(input_h, *self.thetas)

            cost = inhibitory_cost(targetmean2, targetvar2, W, f, **self.dynamics_kwargs)

            cost.backward()
            optimiser.step()

        return cost

    def initial_weight_training(
        self,
        init_input: T,
        target_cov: T,
        target_mean: T,
        num_iterations: int,
        repeats_per_iteration: int,
        num_patterns: int,
        dt: float,
    ):
        num_neurons = init_input.shape[-1] * 2

        all_costs = torch.full((num_iterations, repeats_per_iteration), torch.nan).to(device)
        all_ws = torch.full((num_iterations, repeats_per_iteration, num_neurons, num_neurons), torch.nan).to(device)
        all_thetas = torch.full((num_iterations, repeats_per_iteration, 3), torch.nan).to(device)

        scale_upper = 0 # scale_upper = 2.0
        test_thetas_scales = 0 # test_thetas_scales = torch.tensor([0.1, 0.1, 1.0])
        test_w_scale = 0 # test_w_scale = 0.1

        num_trials = 200
        num_simulation_steps = 1000
        num_burn_in_step = 1500

        for j in range(num_iterations):

            num_neurons = self.num_initial_e_neurons * 2

            currentcost = torch.tensor(torch.nan).to(device)

            counter = 0

            while torch.isnan(currentcost):

                scale = torch.rand([]) * scale_upper
                inhvar = (scale ** 2) * torch.diagonal(target_cov, 0, 2)
                inhmean = scale * target_mean

                self.load_W((torch.randn([num_neurons, num_neurons]) * test_w_scale).to(device))
                self.load_thetas((torch.randn(3) * test_thetas_scales).to(device))
                import pdb; pdb.set_trace()

                # # Added this in myself!
                # self.train_init_w_with_inh(
                #     input_h=init_input,
                #     target_cov=target_cov,
                #     target_mean=target_mean,
                #     inhmean=inhmean, 
                #     inhvar=inhvar
                # )

                inhmean, inhvar, currentcost = self.update_inh(
                    init_input=init_input,
                    num_trials=num_trials,
                    target_mean=target_mean,
                    target_cov=target_cov,
                    num_steps=num_simulation_steps,
                    num_patterns=num_patterns,
                    num_burn_in_step=num_burn_in_step,
                    dt=dt,
                )

                counter += 1
                if torch.isnan(currentcost):
                    print(f"Unstable after {counter} attempt(s)", end='\r')

            iteration_inhmean, iteration_inhvar, current_cost = [_t.detach() for _t in (inhmean, inhvar, currentcost)]


            all_costs[j, 0] = current_cost.detach().item()
            all_ws[j, 0] = self.W.detach().clone()
            all_thetas[j, 0] = self.thetas.detach().clone()

            print("\n")
            print(f"Initialisation {j} found after {counter} attempt(s), cost of {current_cost}! Begining {repeats_per_iteration} training repeats")

            # i.e. now, we have established that this random W is stable, so we can repeat
            for i in range(1, repeats_per_iteration):
                
                self.train_init_w_with_inh(
                    input_h=init_input,
                    target_cov=target_cov,
                    target_mean=target_mean,
                    inhmean=iteration_inhmean, 
                    inhvar=iteration_inhvar
                )
                
                inhmean, inhvar, rep_cost = self.update_inh(
                    init_input=init_input,
                    num_trials=num_trials,
                    target_mean=target_mean,
                    target_cov=target_cov,
                    num_steps=num_simulation_steps,
                    num_patterns=num_patterns,
                    num_burn_in_step=num_burn_in_step,
                    dt=dt,
                )

                print('\tRep cost:', rep_cost.item())

                if torch.isnan(rep_cost):
                    break
                if rep_cost < current_cost:
                    current_cost = rep_cost

                all_costs[j, i] = rep_cost.detach().item()
                all_ws[j, i] = self.W.detach().clone()
                all_thetas[j, i] = self.thetas.detach().clone()

        return all_costs, all_ws, all_thetas


    def init_dynamics(self, num_trials: int, num_patterns: int):
        num_neurons = self.get_num_neurons()
        u0 = torch.zeros([num_patterns, num_neurons, num_trials]).to(device)
        return u0

    def tau_vector(self, num_trials: int):
        num_neuron_of_each_type = self.get_num_neurons() // 2
        return torch.concat(
            [
                torch.ones(num_neuron_of_each_type) * self.dynamics_kwargs["tau_e"],
                torch.ones(num_neuron_of_each_type) * self.dynamics_kwargs["tau_i"],
            ]
        ).unsqueeze(-1).repeat(1, 1, num_trials).to(device)

    def new_eta(self, num_trials: int, num_patterns: int, old_eta=None):
        num_neurons = self.get_num_neurons()
        new_rand = self.N_matrix @ torch.randn([num_patterns, num_neurons, num_trials]).to(device)
        if old_eta is not None:
            old_comp = old_eta * self.dynamics_kwargs['eps1']
            new_comp = self.dynamics_kwargs['eps2'] * new_rand
            return (old_comp + new_comp).to(device)
        else:
            return (new_rand).to(device)

    def run_dynamics(self, num_trials: int, num_patterns: int, num_steps: int, dt: float, h: T, num_burn_in_steps = 0):
        
        eta = self.new_eta(num_trials, num_patterns)            # [patterns, total neurons, trials]
        u = self.init_dynamics(num_trials, num_patterns)        # [patterns, total neurons, trials]
        w = self.mask_weight()                                  # [total neurons, total neurons]
        tau = self.tau_vector(num_trials)                       # [1 (?), total neurons, trials]
        f = input_nonlinearity(h, *self.thetas).repeat(1, 1, num_trials).float()
        
        self.eval()
        for _ in range(num_burn_in_steps):
            du_dt = dynamics_step(
                u=u, h=h, W=w, eta=eta, tau=tau, f=f, **self.dynamics_kwargs
            )
            u = u + (du_dt * dt)
            eta = self.new_eta(num_trials, num_patterns, eta)

        u_history = []

        self.train()
        for _ in range(num_steps):
            du_dt = dynamics_step(
                u=u, h=h, W=w, eta=eta, tau=tau, f=f, **self.dynamics_kwargs
            )
            u = u + (du_dt * dt)
            u_history.append(u)
            eta = self.new_eta(num_trials, num_patterns, eta)

        return u_history
