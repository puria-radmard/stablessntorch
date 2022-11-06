import torch
import torch.nn.functional as F
from torch import Tensor as T, unsqueeze
from tqdm import tqdm

import matplotlib.pyplot as plt

SCALE_LIMITS = (0.2, 1.0)
SCALE_TO_SIGMA = 5
SCALE_TO_K = 0.5
GAMMA = 1
SQUARE_SIZE = 32

BATCH_SIZE = 1000


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def gsm_forward_pass(filter_set: T, latent_values: T, noise_term: T, contrast) -> T:
    return contrast * (filter_set @ latent_values) + noise_term

def gabor(theta: T, scale: T, _x: T, _y: T, square_size: int) -> T:
    """
    "Forward-pass" of a set of filter parameters (theta_A) to the actual filter set A
    """

    theta = torch.clip(theta, -torch.pi, torch.pi)
    scale = torch.clip(scale, *SCALE_LIMITS)
    _x = torch.clip(_x, -square_size / 2, square_size / 2)
    _y = torch.clip(_y, -square_size / 2, square_size / 2)

    num_filters = _y.shape[0]
    sigma = SCALE_TO_SIGMA * scale
    k = SCALE_TO_K / scale
    gamma = GAMMA

    # Both are [square_size, square_size]
    canvas_x, canvas_y = torch.meshgrid(
        torch.arange(-square_size / 2, square_size / 2).to(device),
        torch.arange(-square_size / 2, square_size / 2).to(device),
    )

    # Both are [square_size, square_size, num_filters]
    canvas_x_offset = canvas_x.unsqueeze(-1).repeat(1, 1, num_filters) - _x
    canvas_y_offset = canvas_y.unsqueeze(-1).repeat(1, 1, num_filters) - _y

    # Both are [square_size, square_size, num_filters]
    axis_1 = canvas_x_offset * torch.cos(theta) + canvas_y_offset * torch.sin(theta)
    axis_2 = canvas_y_offset * torch.cos(theta) - canvas_x_offset * torch.sin(theta)

    gauss = torch.exp(
        -((gamma ** 2) * (axis_1 ** 2) + (axis_2 ** 2)) / (2 * (sigma ** 2))
    )
    sinusoid = torch.cos(k * axis_1)
    fields = gauss * sinusoid

    return fields, gauss, sinusoid


def ols_fit(filter_set: T, image_set: T) -> T:
    # filter set: [full image size, num_filters]
    # image_set: [num_images, full image size]
    # output: [num_images, num_filters]
    left_inv = torch.linalg.inv( filter_set.T @ filter_set ) @ filter_set.T
    return (left_inv @ image_set.unsqueeze(-1)).squeeze()


def ols_projection(filter_set: T, image_set: T) -> T:
    "Appendix A, equation 2"
    # filter set: [full image size, num_filters]
    # image_set: [num_images, full image size]
    # output: [num_images, full image size]
    latents = ols_fit(filter_set, image_set)
    return torch.stack([(lt * filter_set).sum(1) for lt in latents])


def unexplained_variance_loss(x: T, x_ols: T, mean = True) -> T:
    "Appendix A, equation 3. Assume all of size [n_filters, n_pixels]"
    error_squared = torch.square(x - x_ols).sum(-1)
    image_power = torch.square(x).sum(-1)
    fvu = (error_squared / image_power)
    return fvu.mean() if mean else fvu


def train_gsm_projective_fields(
    thetas: T,
    scales: T,
    x_mids: T,
    y_mids: T,
    image_set: T,
    square_size: int,
    n_iter: int,
) -> T:
    """
    Optimise theta_A as per Appendix A, equation 3.
    Currently don't return theta_A, just the final A that they use
    """
    params = [thetas, scales, x_mids, y_mids]
    optimiser = torch.optim.Adam(lr=0.01, params=params)

    history = []

    print("Training GSM projective fields")
    # for i in tqdm(range(n_iter)):
    for i in tqdm(range(n_iter)):

        optimiser.zero_grad()

        # [square_size, square_size, num_filters]
        projective_fields, gauss, sinusoid = gabor(theta=thetas, scale=scales, _x=x_mids, _y=y_mids, square_size=square_size)

        # if i%10 == 0:
        #     fig, axes = plt.subplots(5, 3)
        #     [ax.imshow(projective_fields[:,:,i].cpu().detach()) for i, ax in enumerate(axes[:,0])]
        #     [ax.imshow(gauss[:,:,i].cpu().detach()) for i, ax in enumerate(axes[:,1])]
        #     [ax.imshow(sinusoid[:,:,i].cpu().detach()) for i, ax in enumerate(axes[:,2])]
        #     fig.savefig('asdf.png')

        # [square_size * square_size, num_filters]
        projective_fields = projective_fields.reshape(square_size * square_size, -1)

        x_ols = ols_projection(projective_fields, image_set)
        V_loss = unexplained_variance_loss(image_set, x_ols)

        V_loss.backward()
        optimiser.step()

        history.append(V_loss.item())
        print(history[-1])


    final_projective_fields, _, _ = gabor(
        theta=thetas, scale=scales, _x=x_mids, _y=y_mids, square_size=square_size
    )
    final_projective_fields = final_projective_fields.detach()

    return final_projective_fields, history



def _log_p_x_given_z_gauss(A: T, C: T, z: T, sigma_x: T) -> T:
    # Avoid repeated calcs that don't depend on X

    Nz = z.shape[0]
    filter_size = A.shape[0]

    # [total filter size, total filter size, Nz]
    ACAT = (A @ C @ A.T).unsqueeze(0).repeat(Nz, 1, 1)
    eye = torch.eye(filter_size).to(device)
    noise_term = (torch.square(sigma_x) * eye).unsqueeze(0).repeat(Nz, 1, 1)
    x_given_z_covar = ((z.reshape(-1, 1, 1) ** 2) * (ACAT)) + noise_term

    # [Nz, total filter size]
    zero = torch.zeros(x_given_z_covar.shape[:-1]).to(device)

    x_given_z_covar_chol = torch.linalg.cholesky(x_given_z_covar)
    gauss = torch.distributions.MultivariateNormal(zero, scale_tril=x_given_z_covar_chol)

    return gauss


def log_p_x_given_z(xs: T, A: T = None, C: T = None, z: T = None, sigma_x: T = None, gauss = None) -> T:

    if gauss is None:
        gauss = _log_p_x_given_z_gauss(A=A, C=C, z=z, sigma_x=sigma_x)

    batch_xs = xs.unsqueeze(1)
    
    # [Nz, num images]
    gaussian_loglikelihood = gauss.log_prob(batch_xs)

    return gaussian_loglikelihood


def log_p_z(z: T, alpha: T, beta: T) -> T:
    gamma_likelihood = torch.distributions.gamma.Gamma(
        concentration=alpha, rate=beta
    ).log_prob(z)
    return gamma_likelihood


def p_y_given_x_z(z: T, sigma_x: T, A: T, C: T, x: T):
    Nz = z.shape[0]

    # [num_filters, num_filters, Nz]
    Cinv = torch.linalg.inv(C).unsqueeze(-1).repeat(1, 1, Nz)

    # [num_filters, num_filters, Nz]
    AtA = (A.T @ A).unsqueeze(-1).repeat(1, 1, Nz)

    # [Nz]
    coeff = (z / sigma_x) ** 2  

    # [num_filters, num_filters, Nz] -> changed
    S_z = torch.linalg.inv((Cinv + (coeff * AtA)).permute(-1, 0, 1))  

    # [num_images, num_filters, Nz]
    mu_z = (coeff / z) * ((S_z @ A.T) @ x.T).permute(2, 1, 0)  
    
    return mu_z, S_z.permute(2, 1, 0)


def train_gsm_bayesian_parameters(
    image_set: T,
    filter_set: T,
    latent_prior_covar_cholesky: T,
    log_pixel_noise_var: T,
    log_contrast_alpha: T,
    log_contrast_beta: T,
    n_iter: int,
    Nz: int,
):
    """
    Optimise the remaining probabilistic parameters of the GSM, as per Appendix A, equation 4.
    """
    params = [
        latent_prior_covar_cholesky,
        log_pixel_noise_var,
        log_contrast_alpha,
        log_contrast_beta,
    ]
    optimiser = torch.optim.Adam(lr=0.1, params=params)

    history = []

    for _ in tqdm(range(n_iter)):

        zs = torch.linspace(0.01, 10, Nz).to(device)

        for b in range(0, image_set.shape[0], BATCH_SIZE):

            CL_ = torch.tril(latent_prior_covar_cholesky)
            C = CL_ @ CL_.T

            # [Nz]
            prior_log_p_z = log_p_z(z=zs, alpha=log_contrast_alpha.exp(), beta=log_contrast_beta.exp())

            batch_image_set = image_set[b:b+BATCH_SIZE].to(device)

            p_x_given_z_gauss = _log_p_x_given_z_gauss(
                A=filter_set,
                C=C,
                z=zs,
                sigma_x=log_pixel_noise_var.exp(),
            )

            # [num_images, Nz]
            conditional_log_p_x_given_z = log_p_x_given_z(
                xs = batch_image_set,
                gauss = p_x_given_z_gauss
            )

            # Product and integrate over all zs and sum over all images
            # However for numerical stability we need to do a minus baseline

            # [num_images, Nz]
            joint_log_likelihood = conditional_log_p_x_given_z + prior_log_p_z
            joint_log_likelihood_baseline: T = joint_log_likelihood.max(-1).values
            
            # Marginalise
            negative_llh_baseline_removed = (joint_log_likelihood - joint_log_likelihood_baseline.unsqueeze(-1)).exp().sum(-1)
            negative_llh = - (negative_llh_baseline_removed.log() + joint_log_likelihood_baseline).mean()
            
            # Compound loss
            total_negative_llh = negative_llh
            
            torch.cuda.empty_cache()
            total_negative_llh.backward()
            optimiser.step()

        # total_negative_llh.backward()

        assert not any([p.isnan().any() for p in params])
        assert not any([p.grad.isnan().any() for p in params])

        history.append(negative_llh.item())

    return history


def gsm_inference(
    image_set: T,
    projective_fields: T,
    log_pixel_noise_var: T,
    latent_prior_covar_cholesky: T,
    log_contrast_alpha: T,
    log_contrast_beta: T,
    Nz: int,
):

    zs = torch.linspace(0.01, 1.5, Nz).to(device)

    # [num_filter, num_filters]
    prior_C = latent_prior_covar_cholesky @ latent_prior_covar_cholesky.T

    # [num_images, Nz]
    log_joint_x_z = (
        # [num_images, Nz]
        log_p_x_given_z(xs=image_set, A=projective_fields, C=prior_C, z=zs, sigma_x=log_pixel_noise_var.exp())  
        # [Nz], no repeat needed
        + log_p_z(z=zs, alpha=log_contrast_alpha.exp(), beta=log_contrast_beta.exp())  
    )
    log_joint_x_z_baseline: T = log_joint_x_z.max(-1).values

    # Not marginalising this time
    joint_x_z_baseline_removed = (log_joint_x_z - log_joint_x_z_baseline.unsqueeze(-1)).exp()

    # [num_images, Nz]
    p_z_given_x = joint_x_z_baseline_removed / joint_x_z_baseline_removed.sum(axis=1).unsqueeze(-1)

    # conditional_p_y_given_x_z_mean:  [num_images, num_filters, Nz]
    # conditional_p_y_given_x_z_covar: [num_filters, num_filters, Nz]   - same for all x
    conditional_p_y_given_x_z_mean, conditional_p_y_given_x_z_covar = p_y_given_x_z(
        z=zs, sigma_x=log_pixel_noise_var.exp(), A=projective_fields, C=prior_C, x=image_set
    )

    # [num_images, num_filters] <- sum over z axis
    latent_means = (conditional_p_y_given_x_z_mean * p_z_given_x.unsqueeze(1)).sum(-1)

    # [num_images, num_filters, Nz]
    mean_spread = conditional_p_y_given_x_z_mean - latent_means.unsqueeze(-1)

    # [num_images, num_filters, num_filters, Nz]
    mean_spread_outer = torch.einsum("bia,bja->bija", (mean_spread, mean_spread))

    # [num_images, num_filters, num_filters]
    latent_vars = (
        (conditional_p_y_given_x_z_covar + mean_spread_outer)
        * p_z_given_x.unsqueeze(1).unsqueeze(1)
    ).sum(-1)

    assert not latent_means.isnan().any()
    assert not latent_vars.isnan().any()

    return latent_means, latent_vars
