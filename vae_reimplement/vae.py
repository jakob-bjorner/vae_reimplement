# implementing some sort of variational autoencoding model
import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Any

class VAE(nn.Module):
    def __init__(self, prior_dim: int, prior_epsilon: float, beta_weight_on_kl: float, 
                 image_HW: int = None, image_channels: int = None, 
                 encoder: nn.Module=None, decoder: nn.Module=None, 
                 reconstruction_type: str=None, assume_ones_image_std=True):
        """
        Parameters:
            prior_dim (int): the size of the latent, which will be used for the image model
            prior_epsilon (int): the sigma value used to augment the N(0,1), which is typically used for the prior
            beta_weight_on_kl (float):
            encoder (nn.Module):
            decoder (nn.Module):
            
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``
        """
        super(VAE, self).__init__()
        
        assert (image_HW is not None and image_channels is not None) or (encoder is not None and decoder is not None)
        self.prior_epsilon = prior_epsilon
        self.beta_weight_on_kl = beta_weight_on_kl
        self.prior_dim = prior_dim
        self.image_channels = image_channels
        self.image_HW = image_HW
        self.reconstruction_type = "LOGPROB" if reconstruction_type is None else reconstruction_type
        self.assume_ones_image_std = assume_ones_image_std

        self.encoder = encoder if encoder is not None else nn.Sequential(nn.Conv2d(image_channels, 2 * prior_dim, (image_HW, image_HW)), nn.Flatten())
        self.decoder = decoder if decoder is not None else nn.Sequential(nn.Linear(prior_dim, 2* image_HW * image_HW * image_channels))

    def forward(self, image):
        # given some input x, produce a latent from your encoder, and then applying the KL divergence loss with this, train the model in an MCMC fashion.
        
        # model outputs the log because then it can output negatives and we just convert that to something close to zero.
        posterior_q_z_give_x_mu, posterior_q_z_give_x_sigma_2_log = self.encoder(image).chunk(2, dim=-1)
        
        posterior_dist = torch.distributions.normal.Normal(posterior_q_z_give_x_mu, torch.exp(0.5 * posterior_q_z_give_x_sigma_2_log))
        sample_latent = posterior_dist.rsample()

        image_x_give_z_mu, image_x_give_z_sigma_2_log = self.decoder(sample_latent).chunk(2, dim=-1)
        data_dist = torch.distributions.normal.Normal(image_x_give_z_mu, torch.ones_like(image_x_give_z_sigma_2_log) if self.assume_ones_image_std else torch.exp(0.5 * image_x_give_z_sigma_2_log))

        return dict(data_dist=data_dist, posterior_dist=posterior_dist, sample_latent=sample_latent)

        # simga isn't used from the image for sampling only for los calculation.
        # find the prob of the image given under the output of the model.

    def loss(self, image):
        ret_dict = self.forward(image)
        data_dist, posterior_dist, sample_latent = ret_dict["data_dist"], ret_dict["posterior_dist"], ret_dict["sample_latent"]
        prior_dist = torch.distributions.normal.Normal(torch.zeros_like(sample_latent), torch.ones_like(sample_latent)* self.prior_epsilon)
        # .sample does the Reinforce trick: delt Ex[f(x)] -> Ex[f(x) * delt log(f(x))], where x sampled from f(x).
        # .rsample is using the reparameterization like from vae paper

        # try reconstruction loss
        reconstruction_losses = self._get_reconstruction_loss(data_dist, image)
        # print(reconstruction_loss)
        reconstruction_loss = reconstruction_losses["reconstruction_loss"]
        kl_term = torch.distributions.kl_divergence(posterior_dist, prior_dist).flatten(1).sum(axis=1)
        # print("kl_term", kl_term)
        beta_times_kl = self.beta_weight_on_kl * kl_term
        
        # loss = (-log_prob_sample + beta_times_kl).mean()
        loss = (reconstruction_loss + beta_times_kl).mean()
        # import pdb; pdb.set_trace()
        elbo = (reconstruction_losses["negative_log_prob"] + kl_term).mean()
        return {"loss": loss, "elbo": elbo, "log_det_std": reconstruction_losses["log_det_std"].mean()}
    
    def _get_reconstruction_loss(self, image_distribution, true_image):
        # breakpoint() this is something I want to try using breakpoints and then debugging the cell in question for easier devlopment with the debug console's autocomplete.
        reconstruction_loss = None
        Batch, *dims = true_image.shape
        true_image = true_image.view(Batch, -1)
        d = np.prod(dims)
        # get the distribution error from samples created from the image_distribution. This can be done using a number of different examples, and their mean error.
        # this consistutes another hyper parameter.
        # Also, should try implementing the math her manually to see the derivation again:
        # where do I start from? the log(p(x)) 
        # = log(sum_z{p(x, z)}) get a latent, because we essentially make an assumption on the creation process for the data, and want to model it's generation in that assumption.
        # = log(E_{z~q{z|x)}{p(x, z) / q(z | x)})next, we want to inform the process by which we construct the z, as trying to do this marginalization would be impractical, so use a q(z | x), from which we will sample from to inform the learning process.
        # >= E_{z~q{z|x)}{log(p(x|z)) - log(q(z|x) / p(z))}The preceeding equaiton takes many forms, but first we move the log into the expectation resulting in an inequality from jensen's. this to make tractable terms??? ratio of probs is intractible yes?
        # the left term is something we sample, and the right to the KL.
        # here is where we are breaking down the left term's loss, as it represents the reconstruction step, and a term we want to optimize to make the p(x|z) decoder effective.
        # E_{z~q{z|x)}{log(p(x|z))}, we are looking at the probability of the true data under the distribution given by the sampled z's.
        # ln( 1/(sqrt(2*pi)*sigma) * exp(-1/2 * (x-mu)^2 / sigma^2) )under a single normal distribution we see the log of this contains an mse term along with some sigmoid terms.
        # -1/2 * ln(2pi) - ln(sigma) - 1/2 * (x-mu)^2 / sigma^2, when the derivative is taken it is different from just MSE because of these sigmoid terms, so it is fair to check that they are the same in performance, and should check speed of everything as well.
        negative_log_prob = d/2 * np.log(2* np.pi) + image_distribution.scale.log().sum(1) + 1/2 * (((true_image - image_distribution.loc)/image_distribution.scale)**2).sum(1)
        # assert torch.allclose(negative_log_prob, -image_distribution.log_prob(true_image).sum(1))
        if self.reconstruction_type == "LOGPROB":
            reconstruction_loss = negative_log_prob
        elif self.reconstruction_type == "MSE":
            reconstruction_loss = torch.nn.functional.mse_loss(image_distribution.loc, true_image, reduction="none").sum(1)
        else:
            raise Exception("invalid reconstruction type specified. Only LOGPROB or MSE!", self.reconstruction_type)
        return {"reconstruction_loss": reconstruction_loss, "negative_log_prob": negative_log_prob, "log_det_std": image_distribution.scale.log().sum(1)} 
             # Get into hydra, and get the development working so you can try a bunch of things.
    def sample_images_from_latent(self, images):
        with torch.no_grad():
            ret_dict = self.forward(images)
            data_dist = ret_dict['data_dist']
            # data_dist = torch.distributions.normal.Normal(image_x_give_z_mu, torch.exp(0.5 * image_x_give_z_sigma_log))
            # images = data_dist.rsample().chunk(len(images), dim=0)
            return data_dist.loc.reshape(-1, self.image_channels, self.image_HW, self.image_HW)
    def sample_images(self, num_samples, device='cuda'):
        priors = torch.randn((num_samples, self.prior_dim), dtype=torch.float32, device=device) * self.prior_epsilon
        with torch.no_grad():
            image_x_give_z_mu, image_x_give_z_sigma_log = self.decoder(priors).chunk(2, dim=-1)
            # data_dist = torch.distributions.normal.Normal(image_x_give_z_mu, torch.exp(0.5 * image_x_give_z_sigma_log))
            # images = data_dist.rsample().chunk(num_samples, dim=0)
            return image_x_give_z_mu.reshape(-1, self.image_channels, self.image_HW, self.image_HW) # this image should depend on the transforms used to make it.

if __name__ == "__main__":
    model = VAE(prior_dim=10, prior_epsilon=1, beta_weight_on_kl=1, image_HW=32, image_channels=3, reconstruction_type="MSE").to("cuda")

    print(model.loss(torch.ones([2, 3, 32, 32]).cuda()))