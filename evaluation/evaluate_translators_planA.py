import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_vae import VAE
from training.train_translators_planA_adv import TranslatorMLP 


def collect_latents(vae: VAE, loader: DataLoader, device: str, max_samples: int):
    """Encode up to max_samples images into latent means."""
    vae.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, logvar = vae.encode(x)
            latents.append(mu.cpu())
            labels.append(y.clone())
            if sum(t.size(0) for t in latents) >= max_samples:
                break
    z = torch.cat(latents, dim=0)
    y_all = torch.cat(labels, dim=0)
    if z.size(0) > max_samples:
        z = z[:max_samples]
        y_all = y_all[:max_samples]
    return z, y_all  # (N, M), (N,)


def pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances: z (N, M) -> D (N, N)."""
    diff = z.unsqueeze(1) - z.unsqueeze(0)  # (N, N, M)
    D = torch.norm(diff, dim=-1)           # (N, N)
    return D


def upper_triangle_flat(D: torch.Tensor) -> torch.Tensor:
    """Flatten upper triangular part (excluding diagonal) into a 1D vector."""
    N = D.size(0)
    iu = torch.triu_indices(N, N, offset=1)
    return D[iu[0], iu[1]]  # (N*(N-1)/2,)


def pearson_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two 1D tensors."""
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x ** 2).sum() * (y ** 2).sum())
    if den == 0:
        return 0.0
    return (num / den).item()


def knn_indices(z: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of k nearest neighbors (excluding self) for each point."""
    with torch.no_grad():
        D = pairwise_distances(z)          # (N, N)
        N = D.size(0)
        # avoid picking self as neighbor
        D[torch.arange(N), torch.arange(N)] = float("inf")
        knn = torch.topk(D, k, dim=1, largest=False).indices
    return knn  # (N, k)


def knn_overlap(z_orig: torch.Tensor, z_mapped: torch.Tensor, k: int) -> float:
    """Average fraction of shared neighbors between orig and mapped spaces."""
    idx_orig = knn_indices(z_orig, k=k)      # (N, k)
    idx_mapped = knn_indices(z_mapped, k=k)  # (N, k)

    N = idx_orig.size(0)
    overlaps = []
    for i in range(N):
        set_o = set(idx_orig[i].tolist())
        set_m = set(idx_mapped[i].tolist())
        inter = len(set_o.intersection(set_m))
        overlaps.append(inter / k)
    return float(sum(overlaps) / len(overlaps))


def compute_class_means(z: torch.Tensor, y: torch.Tensor, num_classes: int = 10):
    """Compute mean latent vector for each class k in {0,...,num_classes-1}."""
    means = []
    for k in range(num_classes):
        mask = (y == k)
        if mask.sum() == 0:
            # shouldn't happen with MNIST, but be safe
            means.append(torch.zeros(z.size(1)))
        else:
            means.append(z[mask].mean(dim=0))
    return torch.stack(means, dim=0)  # (num_classes, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--weights-vaeA", type=str, default="checkpoints/mnist_split_vaeA.pt")
    ap.add_argument("--weights-vaeB", type=str, default="checkpoints/mnist_split_vaeB.pt")
    ap.add_argument("--weights-TAB", type=str, default="checkpoints/mnist_split_T_AB.pt")
    ap.add_argument("--weights-TBA", type=str, default="checkpoints/mnist_split_T_BA.pt")
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--k-nn", type=int, default=10)
    args = ap.parse_args()

    device = args.device

    # Load VAEs
    vaeA = VAE(latent_dim=args.latent_dim)
    vaeA.load_state_dict(torch.load(args.weights_vaeA, map_location=device))
    vaeB = VAE(latent_dim=args.latent_dim)
    vaeB.load_state_dict(torch.load(args.weights_vaeB, map_location=device))
    vaeA.to(device)
    vaeB.to(device)

    # Load translators
    T_AB = TranslatorMLP(args.latent_dim)
    T_AB.load_state_dict(torch.load(args.weights_TAB, map_location=device))
    T_BA = TranslatorMLP(args.latent_dim)
    T_BA.load_state_dict(torch.load(args.weights_TBA, map_location=device))
    T_AB.to(device).eval()
    T_BA.to(device).eval()

    # --- Test set loader (hold-out) ---
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(
        args.data_root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # Collect latents (means) from TEST set for both VAEs
    zA, y_test = collect_latents(vaeA, test_loader, device, max_samples=args.max_samples)
    zB, _      = collect_latents(vaeB, test_loader, device, max_samples=args.max_samples)

    print(f"Collected zA (test): {zA.shape}, zB (test): {zB.shape}")

    # Map latents through translators
    with torch.no_grad():
        zA_mapped = T_AB(zA.to(device)).cpu()
        zB_mapped = T_BA(zB.to(device)).cpu()

    # --- Identity check: is T_AB ~ identity? ---
    with torch.no_grad():
        diffA = (zA_mapped - zA).pow(2).sum(dim=1).sqrt().mean()
        diffB = (zB_mapped - zB).pow(2).sum(dim=1).sqrt().mean()

    print("Mean L2 norm ||T_AB(zA) - zA|| on TEST:", diffA.item())
    print("Mean L2 norm ||T_BA(zB) - zB|| on TEST:", diffB.item())

    # --- Distance correlation (test set) ---
    DA = pairwise_distances(zA)
    DA_m = pairwise_distances(zA_mapped)
    DB = pairwise_distances(zB)
    DB_m = pairwise_distances(zB_mapped)

    DA_flat = upper_triangle_flat(DA)
    DA_m_flat = upper_triangle_flat(DA_m)
    DB_flat = upper_triangle_flat(DB)
    DB_m_flat = upper_triangle_flat(DB_m)

    corr_A = pearson_corr_torch(DA_flat, DA_m_flat)
    corr_B = pearson_corr_torch(DB_flat, DB_m_flat)

    print(f"Distance correlation A (orig vs mapped, TEST): {corr_A:.4f}")
    print(f"Distance correlation B (orig vs mapped, TEST): {corr_B:.4f}")

    # --- k-NN overlap (test set) ---
    overlap_A = knn_overlap(zA, zA_mapped, k=args.k_nn)
    overlap_B = knn_overlap(zB, zB_mapped, k=args.k_nn)

    print(f"{args.k_nn}-NN overlap A (orig vs mapped, TEST): {overlap_A:.4f}")
    print(f"{args.k_nn}-NN overlap B (orig vs mapped, TEST): {overlap_B:.4f}")

    # --- Class-mean alignment test ---
    # Means in A and B (TEST)
    mu_A = compute_class_means(zA, y_test, num_classes=10)        # (10, M)
    mu_B = compute_class_means(zB, y_test, num_classes=10)        # (10, M)

    with torch.no_grad():
        mu_A_to_B = T_AB(mu_A.to(device)).cpu()                   # (10, M)

    # For each digit k, find nearest class in B
    correct = 0
    for k in range(10):
        diffs = mu_B - mu_A_to_B[k]          # (10, M)
        dists = diffs.pow(2).sum(dim=1).sqrt()  # (10,)
        nearest = torch.argmin(dists).item()
        print(f"Digit {k}: nearest in B-space is {nearest} (dist={dists[nearest].item():.4f})")
        if nearest == k:
            correct += 1

    print(f"Class-mean alignment accuracy (A→B): {correct}/10 = {correct/10:.2f}")


if __name__ == "__main__":
    main()
