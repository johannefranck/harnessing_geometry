import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

def plot_prior_and_agg_posterior(
    model, 
    data_loader, 
    device, 
    latent_dim, 
    filename='prior_agg_posterior.png',
    n_prior_samples=5000):
    
    model.eval()
    # -------------------------------
    # 1) Sample from the prior
    # -------------------------------
    with torch.no_grad():
        z_prior = model.prior().sample((n_prior_samples,)).to(device)  # shape: (n_prior_samples, latent_dim)

    # -------------------------------
    # 2) Sample from the aggregated posterior
    # -------------------------------
    z_post_list = []
    labels_list = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            # q(z|x) = model.encoder(x), which is a Gaussian distribution
            qz_x = model.encoder(x_batch)
            # Let's sample (or use .mean if you prefer) to get a single z per x
            z_samps = qz_x.rsample()  # shape: (batch_size, latent_dim)
            z_post_list.append(z_samps.cpu())
            labels_list.append(y_batch)

    # Concatenate across all minibatches
    
    z_post = torch.cat(z_post_list[:100], dim=0)    # shape: (N, latent_dim)
    labels = torch.cat(labels_list[:100], dim=0)    # shape: (N,)

    # -------------------------------
    # 3) Do PCA on combined set of points
    # -------------------------------
    z_prior_np = z_prior.cpu().numpy()
    z_post_np  = z_post.numpy()

    combined = np.concatenate([z_prior_np, z_post_np], axis=0)
    
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)  # shape: (n_prior_samples + N, 2)
    
    z_prior_2d = combined_2d[:n_prior_samples]
    z_post_2d  = combined_2d[n_prior_samples:]

    if latent_dim == 2:
        z_prior_2d = z_prior_np
        z_post_2d = z_post_np
    
    # -------------------------------
    # 4) Make the plot
    # -------------------------------
    kde_fn = gaussian_kde(z_prior_2d.T)

    # Create a grid for contour plotting
    x_min, x_max = z_prior_2d[:,0].min(), z_prior_2d[:,0].max()
    y_min, y_max = z_prior_2d[:,1].min(), z_prior_2d[:,1].max()
    
    X, Y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    Z = kde_fn(grid_points).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.contourf(X, Y, Z, levels=20, vmin=Z.min(), vmax=Z.max(), cmap="Purples")# cmap="Purples"
    ax.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5)

    sc = ax.scatter(
        z_post_2d[:,0], z_post_2d[:,1], 
        c=labels, cmap="tab10", s=2, alpha=0.8,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_aggregate_posterior(model, data_loader, device, latent_dim, filename):
    """
    Plot samples from the approximate posterior (aggregate posterior) 
    and color them by their correct class label. For latent dimensions > 2,
    the latent samples are projected onto the first two principal components using PCA.
    
    Parameters:
        model: [VAE]
            The trained VAE model.
        data_loader: [torch.utils.data.DataLoader]
            The data loader for the test set.
        device: [torch.device]
            The device to run inference on.
        latent_dim: [int]
            The dimensionality of the latent space.
    """
    model.eval()
    z_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            # MNIST returns (image, label)
            images, labels = batch[0].to(device), batch[1]
            # Get the approximate posterior q(z|x) from the encoder
            q = model.encoder(images)
            z = q.rsample()  
            z_list.append(z.cpu())
            labels_list.append(labels.cpu())
    
    # Concatenate all batches into one tensor
    z_all = torch.cat(z_list, dim=0)  # shape: (num_samples, latent_dim)
    labels_all = torch.cat(labels_list, dim=0)  # shape: (num_samples,)
    
    # If latent_dim > 2, project the latent samples onto 2D using PCA
    if latent_dim > 2:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_all.numpy())
    else:
        # If latent_dim is exactly 2, no projection is needed.
        z_2d = z_all.numpy()
    
    # Plot using matplotlib
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all.numpy(), cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Aggregate Posterior: Latent Space')
    plt.tight_layout()
    plt.savefig(filename)
    
def plot_prior(
    model, 
    data_loader, 
    device, 
    latent_dim, 
    filename,
    n_prior_samples=5000):
    
    model.eval()
    # -------------------------------
    # 1) Sample from the prior
    # -------------------------------
    with torch.no_grad():
        z_prior = model.prior().sample((n_prior_samples,)).to(device)  # shape: (n_prior_samples, latent_dim)

    # -------------------------------
    # 2) Sample from the aggregated posterior
    # -------------------------------
    z_post_list = []
    labels_list = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            # q(z|x) = model.encoder(x), which is a Gaussian distribution
            qz_x = model.encoder(x_batch)
            # Let's sample (or use .mean if you prefer) to get a single z per x
            z_samps = qz_x.rsample()  # shape: (batch_size, latent_dim)
            z_post_list.append(z_samps.cpu())
            labels_list.append(y_batch)

    # Concatenate across all minibatches
    
    z_post = torch.cat(z_post_list[:100], dim=0)    # shape: (N, latent_dim)
    labels = torch.cat(labels_list[:100], dim=0)    # shape: (N,)

    # -------------------------------
    # 3) Do PCA on combined set of points
    # -------------------------------
    z_prior_np = z_prior.cpu().numpy()
    z_post_np  = z_post.numpy()

    combined = np.concatenate([z_prior_np, z_post_np], axis=0)
    
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)  # shape: (n_prior_samples + N, 2)
    
    z_prior_2d = combined_2d[:n_prior_samples]
    z_post_2d  = combined_2d[n_prior_samples:]

    if latent_dim == 2:
        z_prior_2d = z_prior_np
        z_post_2d = z_post_np
    
    # -------------------------------
    # 4) Make the plot
    # -------------------------------
    kde_fn = gaussian_kde(z_prior_2d.T)

    # Create a grid for contour plotting
    x_min, x_max = z_prior_2d[:,0].min(), z_prior_2d[:,0].max()
    y_min, y_max = z_prior_2d[:,1].min(), z_prior_2d[:,1].max()
    
    X, Y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    Z = kde_fn(grid_points).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.contourf(X, Y, Z, levels=20, vmin=Z.min(), vmax=Z.max(), cmap="Purples")# cmap="Purples"
    ax.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()