import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_vae import VAE
from models.supervised_vae import SupervisedVAE


def stratified_split_indices(labels, seed: int = 0):
    """Split indices into two disjoint halves, roughly balanced per label."""
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


def collect_latents(model, loader, device):
    """
    Run the *trained* VAE encoder once over the given loader
    and return (z_mu, labels).
    """
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)   # SupervisedVAE.encode -> (mu, logvar)
            zs.append(mu.cpu())
            ys.append(y.clone())
    z = torch.cat(zs, dim=0)
    y = torch.cat(ys, dim=0)
    return z, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--prefix", type=str, default="checkpoints/mnist_split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device != "cpu" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)

    # 1) Load MNIST and restrict to digits 1,2,3
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        args.data_root, train=True, download=True, transform=transform
    )

    mask = (full_train.targets == 1) | (full_train.targets == 2) | (full_train.targets == 3)
    filtered_indices = mask.nonzero(as_tuple=False).view(-1)
    full_train = Subset(full_train, filtered_indices)

    # 2) Stratified split A vs B USING THE FILTERED LABELS
    labels_subset = full_train.dataset.targets[full_train.indices]
    idxA, idxB = stratified_split_indices(labels_subset, seed=args.seed)

    split_path = f"{args.prefix}_split_indices.pt"
    torch.save({"idxA": idxA, "idxB": idxB}, split_path)
    print(f"Saved split indices to {split_path}")

    dsA = Subset(full_train, idxA)
    dsB = Subset(full_train, idxB)

    loaderA = DataLoader(dsA, batch_size=args.batch_size, shuffle=True)
    loaderB = DataLoader(dsB, batch_size=args.batch_size, shuffle=True)

    # 3) Train VAE_A
    vaeA = SupervisedVAE(latent_dim=args.latent_dim)
    print("Training VAE_A on domain A...")
    vaeA = train_vae(vaeA, loaderA, device, epochs=args.epochs, lr=args.lr)
    torch.save(vaeA.state_dict(), f"{args.prefix}_vaeA.pt")

    # 4) Train VAE_B
    vaeB = SupervisedVAE(latent_dim=args.latent_dim)
    print("Training VAE_B on domain B...")
    vaeB = train_vae(vaeB, loaderB, device, epochs=args.epochs, lr=args.lr)
    torch.save(vaeB.state_dict(), f"{args.prefix}_vaeB.pt")

    # 5) After training: collect and save latents once (no more encoder_B later)
    #    Use deterministic order (shuffle=False)
    loaderA_eval = DataLoader(dsA, batch_size=args.batch_size, shuffle=False)
    loaderB_eval = DataLoader(dsB, batch_size=args.batch_size, shuffle=False)

    zA, yA = collect_latents(vaeA.to(device), loaderA_eval, device)
    zB, yB = collect_latents(vaeB.to(device), loaderB_eval, device)

    torch.save({"z": zA, "y": yA}, f"{args.prefix}_zA_train.pt")
    torch.save({"z": zB, "y": yB}, f"{args.prefix}_zB_train.pt")
    print("Saved zA/zB train latents.")


if __name__ == "__main__":
    main()
