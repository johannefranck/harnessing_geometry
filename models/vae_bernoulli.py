# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from harnessing_geometry.plots.plot_vae import plot_aggregate_posterior, plot_prior_and_agg_posterior, plot_prior

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
            M: [int] Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M # Dimension of the latent space
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False) # requires_grad=False means they are not trainable; they remain fixed as a standard Gaussian
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class MixtureGaussianPrior(nn.Module):
    def __init__(self, M, num_components):
        """
        Define a mixture of Gaussian prior distribution.
        
        Each component is a multivariate (diagonal) Gaussian over the latent space 
        of dimension M. The mixing probabilities are fixed as uniform.

        Parameters:
            M: [int]
               Dimension of the latent space.
            num_components: [int]
               Number of mixture components.
        """
        super(MixtureGaussianPrior, self).__init__()
        self.M = M
        self.num_components = num_components

        # Fixed uniform mixing probabilities.
        self.register_buffer('mixing_probs', torch.ones(num_components) / num_components)

        # Learnable parameters for each component: mean and log-scale.        
        self.component_means = nn.Parameter(torch.randn(num_components, M))
        self.component_log_stds = nn.Parameter(torch.zeros(num_components, M))

    def forward(self):
        # Create a categorical distribution for the mixture components.
        mix = td.Categorical(self.mixing_probs)
        comp = td.Independent(td.Normal(self.component_means, torch.exp(self.component_log_stds)), 1)
        mog = td.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
        return mog

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1) # Split the output of the encoder network into two parts, along the last dimension
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) # Return a Gaussian distribution with mean and standard deviation given by the encoder network


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2) # Return a Bernoulli distribution with logits given by the decoder network


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
    
    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x) # Encode the input 
        z = q.rsample() # Sample latent variable z
        
        log_recon = self.decoder(z).log_prob(x) # Compute the log-likelihood
        
        # For a standard Gaussian prior, we use the closed-form KL divergence
        if args.prior == 'gaussian':
            kl_div = td.kl_divergence(q, self.prior())
        
        # For mixture of Gaussians or VampPrior, we estimate KL divergence via Monte Carlo
        elif args.prior in ['mog', 'vampprior']:
            log_qz = q.log_prob(z)
            log_pz = self.prior().log_prob(z)
            kl_div = log_qz - log_pz
        
        elbo = torch.mean(log_recon - kl_div, dim=0) # Average the ELBO over the batch 
        
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            loss = model(x)
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', help='prior distribution to use (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--num_components', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--runname', type=str, default='default', help='name of the run')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'threshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    num_components = args.num_components
    
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MixtureGaussianPrior(M, num_components)
    
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)
        
        # Evaluate model on test set
        test_loss = evaluate(model, mnist_test_loader, args.device)
        print(f"Test loss: {test_loss:.4f}")
        # Save in txt file
        with open(f'test_elbo_{args.runname}.txt', 'w') as f:
            f.write(f"Test loss: {test_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), f"{args.runname}.pt")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f"{args.runname}.pt", map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), f"VAE_{args.prior}_{args.runname}_samples.png")
        
        # Plot aggregate posterior
        plot_aggregate_posterior(model, mnist_test_loader, args.device, args.latent_dim, f"{args.runname}_aggpost_plot.png")
        
        # Now do the aggregate posterior and prior plot
        plot_prior_and_agg_posterior(
            model=model,
            data_loader=mnist_test_loader, 
            device=device,
            latent_dim=args.latent_dim,
            filename=f"{args.runname}_prior_aggpost_plot.png",
            n_prior_samples=5000
        )
        
        plot_prior(
            model=model,
            data_loader=mnist_test_loader, 
            device=device,
            latent_dim=args.latent_dim,
            filename=f"{args.runname}_prior_plot.png",
            n_prior_samples=5000
        )
    
    
# python vae_bernoulli.py --mode train --prior mog --device cpu --batch-size 32 --epochs 10 --num_components 4 --runname mog_4comp
# python vae_bernoulli.py --mode sample --prior mog --device cpu --batch-size 32 --epochs 10 --num_components 10 --latent-dim 10 --runname vae_mog_10comp_b32_e50_ld10