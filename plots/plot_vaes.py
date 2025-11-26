import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.supervised_vae import SupervisedVAE


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_filtered_mnist_test(data_root: str) -> Subset:
    """
    Load MNIST test set and keep only digits 1, 2, 3.
    Returns a Subset over the test dataset.
    """
    transform = transforms.ToTensor()
    test = datasets.MNIST(
        data_root,
        train=False,
        download=True,   # safe: won't re-download if already present
        transform=transform,
    )

    mask = (test.targets == 1) | (test.targets == 2) | (test.targets == 3)
    idx = mask.nonzero(as_tuple=False).view(-1)
    return Subset(test, idx)

def encode_sample(mu, logvar):
    """Reparameterization trick (your VAE structure)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def get_latent_2d(model, data_loader, device, latent_dim: int):
    """
    Run data through the encoder, gather latent samples z, and
    return a 2D embedding (via PCA if latent_dim > 2) plus labels.
    """

    model.eval()
    z_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            mu, logvar = model.encoder(x)   
            z = encode_sample(mu, logvar)    # (batch, latent_dim)

            z_list.append(z.cpu())
            labels_list.append(y.cpu())

    z_all = torch.cat(z_list, dim=0)       # (N, latent_dim)
    labels_all = torch.cat(labels_list, 0) # (N,)

    if latent_dim > 2:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_all.numpy())  # (N, 2)
    else:
        z_2d = z_all.numpy()

    return z_2d, labels_all.numpy()


def main():
    CHECKPOINT_PREFIX = "checkpoints/mnist_split"
    PLOTS_DIR = "plots"
    BATCH_SIZE = 256
    LATENT_DIM = 8        # must match what you trained with
    DEVICE = "cpu"
    DATA_ROOT = "data/"

    ensure_dir(PLOTS_DIR)

    device = torch.device(DEVICE)

    # --------------------------------------------------
    # 1) Test data: MNIST test, digits 1,2,3 only
    # --------------------------------------------------
    test_subset = load_filtered_mnist_test(DATA_ROOT)
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------
    # 2) Load trained VAEs
    # --------------------------------------------------
    vaeA = SupervisedVAE(latent_dim=LATENT_DIM).to(device)
    vaeA.load_state_dict(
        torch.load(f"{CHECKPOINT_PREFIX}_vaeA.pt", map_location=device)
    )

    vaeB = SupervisedVAE(latent_dim=LATENT_DIM).to(device)
    vaeB.load_state_dict(
        torch.load(f"{CHECKPOINT_PREFIX}_vaeB.pt", map_location=device)
    )

    # --------------------------------------------------
    # 3) Collect latent embeddings for both models
    # --------------------------------------------------
    zA, yA = get_latent_2d(vaeA, test_loader, device, LATENT_DIM)
    zB, yB = get_latent_2d(vaeB, test_loader, device, LATENT_DIM)

    # --------------------------------------------------
    # 4) Make side-by-side plot
    # --------------------------------------------------
    # Shared axis limits so shapes are comparable
    x_min = min(zA[:, 0].min(), zB[:, 0].min())
    x_max = max(zA[:, 0].max(), zB[:, 0].max())
    y_min = min(zA[:, 1].min(), zB[:, 1].min())
    y_max = max(zA[:, 1].max(), zB[:, 1].max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    sc1 = axes[0].scatter(
        zA[:, 0],
        zA[:, 1],
        c=yA,      # true labels
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    axes[0].set_title("VAE A (test digits 1,2,3)")
    axes[0].set_xlabel("latent dim 1")
    axes[0].set_ylabel("latent dim 2")

    sc2 = axes[1].scatter(
        zB[:, 0],
        zB[:, 1],
        c=yB,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    axes[1].set_title("VAE B (test digits 1,2,3)")
    axes[1].set_xlabel("latent dim 1")

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Build legends from actual labels, using the SAME colormap/norm as the scatters
    all_labels = np.unique(np.concatenate([yA, yB]))

    cmap = sc1.cmap
    norm = sc1.norm

    legend_handles = []
    for lbl in all_labels:
        color = cmap(norm(lbl))
        legend_handles.append(
            mpatches.Patch(color=color, label=str(int(lbl)))
        )

    for ax in axes:
        ax.legend(handles=legend_handles, title="Digit", loc="upper right")


    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "vaeA_vaeB.png")
    plt.savefig(out_path)
    plt.close(fig)

    print(f"Saved side-by-side latent plot to: {out_path}")


if __name__ == "__main__":
    main()
