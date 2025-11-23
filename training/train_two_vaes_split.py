import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_vae import VAE
from models.supervised_vae import SupervisedVAE


def stratified_split_indices(labels, seed: int = 0):
    """Split indices into two disjoint halves, roughly balanced per label.

    labels: 1D tensor or list of ints (length N)
    Returns: idxA, idxB as lists of indices
    """
    labels = torch.as_tensor(labels)
    num_classes = int(labels.max().item() + 1)
    g = torch.Generator()
    g.manual_seed(seed)

    idxA = []
    idxB = []
    for c in range(num_classes):
        inds = (labels == c).nonzero(as_tuple=False).view(-1)
        perm = inds[torch.randperm(inds.numel(), generator=g)]
        half = perm.numel() // 2
        idxA.append(perm[:half])
        idxB.append(perm[half:])
    idxA = torch.cat(idxA).tolist()
    idxB = torch.cat(idxB).tolist()
    return idxA, idxB


# def train_vae(model: VAE, loader: DataLoader, device: str, epochs: int, lr: float):
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     model.train()
#     for epoch in range(1, epochs + 1):
#         total_loss = 0.0
#         total_recon = 0.0
#         total_kl = 0.0
#         n_batches = 0
#         for x, _ in loader:
#             x = x.to(device)
#             optimizer.zero_grad()
#             loss, recon, kl = model(x)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             total_recon += recon.item()
#             total_kl += kl.item()
#             n_batches += 1
#         print(f"[Epoch {epoch:3d}] "
#               f"Loss={total_loss / n_batches:.4f} "
#               f"Recon={total_recon / n_batches:.4f} "
#               f"KL={total_kl / n_batches:.4f}")
#     return model
def train_vae(model, loader, device, epochs: int, lr: float):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = total_recon = total_kl = total_cls = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss, recon, kl, cls = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_cls += cls.item()
            n_batches += 1

        print(f"[Epoch {epoch:3d}] "
              f"Loss={total_loss/n_batches:.4f} "
              f"Recon={total_recon/n_batches:.4f} "
              f"KL={total_kl/n_batches:.4f} "
              f"CLS={total_cls/n_batches:.4f}")

    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--prefix", type=str, default="mnist_split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device != "cpu" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)

    # Load full MNIST train set once
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)

    # Stratified split into A and B
    labels = full_train.targets  # tensor of shape (60000,)
    idxA, idxB = stratified_split_indices(labels, seed=args.seed)

    # Save indices so we can reuse them later (for translator training)
    split_path = f"{args.prefix}_split_indices.pt"
    torch.save({"idxA": idxA, "idxB": idxB}, split_path)
    print(f"Saved split indices to {split_path}")

    dsA = Subset(full_train, idxA)
    dsB = Subset(full_train, idxB)

    loaderA = DataLoader(dsA, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, pin_memory=False)
    loaderB = DataLoader(dsB, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, pin_memory=False)

    # Train VAE on domain A
    vaeA = SupervisedVAE(latent_dim=args.latent_dim)
    print("Training VAE_A on domain A...")
    vaeA = train_vae(vaeA, loaderA, device, epochs=args.epochs, lr=args.lr)
    pathA = f"{args.prefix}_vaeA.pt"
    torch.save(vaeA.state_dict(), pathA)
    print(f"Saved VAE_A weights to {pathA}")

    # Train VAE on domain B
    vaeB = SupervisedVAE(latent_dim=args.latent_dim)
    print("Training VAE_B on domain B...")
    vaeB = train_vae(vaeB, loaderB, device, epochs=args.epochs, lr=args.lr)
    pathB = f"{args.prefix}_vaeB.pt"
    torch.save(vaeB.state_dict(), pathB)
    print(f"Saved VAE_B weights to {pathB}")


if __name__ == "__main__":
    main()
