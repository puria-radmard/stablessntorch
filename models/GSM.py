from math import log
from typing import List
import numpy as np
import torch
from torch import nn
from torch import Tensor as T
from torch.nn.parameter import Parameter

from utils.gsm import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


N_ITER = 2000 # 10000
N_Z = 100

class GSM(nn.Module):
    def __init__(self, n_filters: int, square_size: int) -> None:

        self.n_filters = n_filters
        self.square_size = square_size

        super(GSM, self).__init__()

        self.thetas = Parameter(torch.rand(self.n_filters))
        self.scales = Parameter(
            (torch.rand(self.n_filters) * (SCALE_LIMITS[1] - SCALE_LIMITS[0]))
            + SCALE_LIMITS[0]
        )
        self.x_mids = Parameter(
            torch.rand(self.n_filters) * self.square_size - (self.square_size / 2)
        )
        self.y_mids = Parameter(
            torch.rand(self.n_filters) * self.square_size - (self.square_size / 2)
        )

        self.log_pixel_noise_var = Parameter(torch.tensor(0.0))
        self.log_contrast_alpha = Parameter(torch.tensor(log(2.0)))
        self.log_contrast_beta = Parameter(torch.tensor(log(0.5)))

        # Completely overwritten later, this is just for loading state dict
        self.latent_prior_covar_cholesky = Parameter(torch.zeros(n_filters, n_filters))

    def train_projective_fields(self, image_set: T) -> List:
        image_set = image_set.float().to(device)
        self.projective_fields, history = train_gsm_projective_fields(
            thetas=self.thetas,
            scales=self.scales,
            x_mids=self.x_mids,
            y_mids=self.y_mids,
            image_set=image_set,
            square_size=self.square_size,
            n_iter=N_ITER,
        )
        return history

    def initialise_bayesian_params(self, image_set: T):
        # [num_images, n_filters]
        latents = self.generate_ssn_inputs(image_set=image_set, make_positive=False).cpu()
        
        # Both are [n_filters, n_filters]
        latent_prior_covar = torch.cov(latents.T)
        self.latent_prior_covar_cholesky = Parameter(torch.linalg.cholesky(latent_prior_covar).to(device))

    def train_bayesian_parameters(self, image_set: T) -> List:
        image_set = image_set.float()
        filter_set = self.projective_fields.reshape(
            self.square_size * self.square_size, -1
        )
        
        history = train_gsm_bayesian_parameters(
            image_set=image_set,
            filter_set=filter_set,
            latent_prior_covar_cholesky=self.latent_prior_covar_cholesky,
            log_pixel_noise_var=self.log_pixel_noise_var,
            log_contrast_alpha=self.log_contrast_alpha,
            log_contrast_beta=self.log_contrast_beta,
            n_iter=N_ITER,
            Nz=N_Z,
        )

        # Not constrained during training!
        self.latent_prior_covar_cholesky = torch.tril(self.latent_prior_covar_cholesky)
        
        return history

    def train_all(self, image_set: T):
        projective_fields_loss_history = self.train_projective_fields(image_set)
        bayesian_parameters_loss_history = self.train_bayesian_parameters(image_set)
        return {
            "projective_fields_loss_history": projective_fields_loss_history,
            "bayesian_parameters_loss_history": bayesian_parameters_loss_history,
        }

    def infer_latent_distribution(self, image_set: T):
        image_set = image_set.float()
        filter_set = self.projective_fields.reshape(
            self.square_size * self.square_size, -1
        )
        with torch.no_grad():
            return gsm_inference(
                image_set=image_set,
                projective_fields=filter_set,
                log_pixel_noise_var=self.log_pixel_noise_var,
                latent_prior_covar_cholesky=self.latent_prior_covar_cholesky,
                log_contrast_alpha=self.log_contrast_alpha,
                log_contrast_beta=self.log_contrast_beta,
                Nz=N_Z,
            )

    def generate_ssn_inputs(self, image_set: T, make_positive: bool):
        image_set = image_set.float().to(device)
        filter_set = self.projective_fields.reshape(
            self.square_size * self.square_size, -1
        )
        with torch.no_grad():
            # [num_images, num_filters]
            inputs = ols_fit(filter_set=filter_set, image_set=image_set)
            if make_positive: 
                inputs = inputs - torch.amin(inputs)
            return inputs
