import torch.nn.functional as F
from torch import Tensor as T
import torch

def input_nonlinearity(h: T, theta_1: float, theta_2: float, theta_3: float):
    # Same input for E and I cells
    exc_input = (theta_1**2) * torch.exp((theta_3**2) * torch.log(h + (theta_2**2)))
    return exc_input.repeat(1, 2).unsqueeze(-1)


def rate_nonlinearity(u: T, k: float, gamma: float):
    return k * F.relu(u) ** gamma


def inhibitory_cost(target_mean, target_var, W, f, k, gamma, **kwargs):
    # Used for initilistion only, so:
    # W : [2, 2]
    return (
        torch.square(
            target_mean - f + 
            (k * W @ ((target_mean**gamma) + target_var))
        ).mean()
    )


def dynamics_step(tau: T, u: T, f: T, W: T, eta: T, k: float, gamma: float, **kwargs):
    """
    Equation (1), with step in u as the subject.
    NB: does not include dt as step size!
    """
    r = rate_nonlinearity(u, k, gamma)
    du_dt = (-u + f + (W @ r) + eta) / tau
    return du_dt


def dales_mask(weights: T, num_exc: int):
    """
    Assumes [E | I] block
    """
    mask = torch.ones_like(weights)
    mask[:, num_exc: ] = -1
    return weights * torch.abs(mask)


def sampling_cost_from_statistics(u_mean: T, target_mean: T, u_cov: T, target_cov: T, lambda_cov: float, lambda_var: float, lambda_mean: float):
    # [patterns, neurons, neurons]
    cov_bias = torch.square(u_cov - target_cov)
    # [patterns, neurons]
    mean_bias = torch.square(u_mean - target_mean)
    return (
        (lambda_cov * cov_bias.mean()) +
        # [patterns, neurons]
        (lambda_var * torch.diagonal(cov_bias, 0, 2).mean()) +
        (lambda_mean * mean_bias.mean())
    )


def full_loss_function(u_hists: T, target_mean: T, target_cov: T, first_n: int, lambda_cov: float, lambda_var: float, lambda_mean: float):
    """
    u_hists: [num_steps, num_patterns, num_neurons, num_trials]
    target_mean: [num_patterns, num_neurons]
    target_cov:  [num_patterns, num_neurons, num_neurons]

    Mean taken along num_steps and num_trials direction, leaving [num_patterns, num_neurons]
    Similarily, cov will be [num_patterns, num_neurons, num_neurons]
    """
    e_u_hists = u_hists[:,:,:first_n,:]
    num_steps, num_patterns, num_neurons, num_trials = e_u_hists.shape
    e_u_mean = e_u_hists.mean(-1).mean(0)

    # Torch doesn't offer a batched covariance unfortunately so this will have to do
    e_u_cov = torch.zeros([num_patterns, num_neurons, num_neurons])
    for step in range(num_steps):
        addition = torch.stack([_t.cov() for _t in e_u_hists[step]]) / float(num_steps)
        e_u_cov += addition.reshape(e_u_cov.shape)

    return sampling_cost_from_statistics(
        u_mean=e_u_mean,
        target_mean=target_mean,
        u_cov=e_u_cov,
        target_cov=target_cov,
        lambda_cov=lambda_cov,
        lambda_var=lambda_var,
        lambda_mean=lambda_mean
    )
