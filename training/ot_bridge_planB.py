import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_vae import VAE


@torch.no_grad()
def collect_latents_and_labels(vae: VAE,
                               dataset: torch.utils.data.Dataset,
                               device: str,
                               batch_size: int,
                               max_samples: int):
    """
    Encode up to max_samples images from dataset into latent means,
    and collect their labels.

    Returns:
        z:  (N, latent_dim)
        y:  (N,)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    vae.eval()
    latents = []
    labels = []
    total = 0
    for x, y in loader:
        x = x.to(device)
        mu, logvar = vae.encode(x)
        latents.append(mu.cpu())
        labels.append(y.clone())
        total += mu.size(0)
        if total >= max_samples:
            break

    z = torch.cat(latents, dim=0)
    y_all = torch.cat(labels, dim=0)
    if z.size(0) > max_samples:
        z = z[:max_samples]
        y_all = y_all[:max_samples]
    return z, y_all  # (N, M), (N,)


def sinkhorn_ot(cost: torch.Tensor,
                epsilon: float,
                max_iters: int = 500,
                tol: float = 1e-6):
    """
    Compute entropic OT coupling with uniform marginals using Sinkhorn.
    cost: (Na, Nb) cost matrix C_{ij}
    epsilon: entropic regularization strength
    Returns: Pi (Na, Nb) OT plan
    """
    device = cost.device
    Na, Nb = cost.shape

    # Uniform marginals
    a = torch.full((Na,), 1.0 / Na, device=device)
    b = torch.full((Nb,), 1.0 / Nb, device=device)

    # Gibbs kernel
    K = torch.exp(-cost / epsilon)  # (Na, Nb)
    K = torch.clamp(K, min=1e-12)

    # scaling vectors
    u = torch.ones(Na, device=device) / Na
    v = torch.ones(Nb, device=device) / Nb

    for _ in range(max_iters):
        u_prev = u.clone()

        # u = a / (K @ v)
        Kv = K @ v
        Kv = torch.clamp(Kv, min=1e-12)
        u = a / Kv

        # v = b / (K^T @ u)
        KTu = K.t() @ u
        KTu = torch.clamp(KTu, min=1e-12)
        v = b / KTu

        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # OT plan Pi = diag(u) K diag(v)
    U = torch.diag(u)
    V = torch.diag(v)
    Pi = U @ K @ V  # (Na, Nb)

    return Pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--weights-vaeA", type=str, default="checkpoints/mnist_split_vaeA.pt")
    ap.add_argument("--weights-vaeB", type=str, default="checkpoints/mnist_split_vaeB.pt")
    ap.add_argument("--split-path", type=str,default="checkpoints/mnist_split_split_indices.pt")
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--sinkhorn-iters", type=int, default=500)
    ap.add_argument("--sinkhorn-tol", type=float, default=1e-6)
    ap.add_argument("--out-file", type=str, default="checkpoints/mnist_ot_planB.pt")
    args = ap.parse_args()

    device = args.device

    # --- Load VAEs ---
    vaeA = VAE(latent_dim=args.latent_dim)
    vaeA.load_state_dict(torch.load(args.weights_vaeA, map_location=device))
    vaeB = VAE(latent_dim=args.latent_dim)
    vaeB.load_state_dict(torch.load(args.weights_vaeB, map_location=device))
    vaeA.to(device)
    vaeB.to(device)

    # --- Load MNIST train set and split indices (same as before) ---
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        args.data_root, train=True, download=True, transform=transform
    )
    split = torch.load(args.split_path)
    idxA = split["idxA"]
    idxB = split["idxB"]
    dsA = Subset(full_train, idxA)
    dsB = Subset(full_train, idxB)

    # --- Collect latent samples and labels from both domains ---
    zA, yA = collect_latents_and_labels(
        vaeA, dsA, device, args.batch_size, max_samples=args.max_samples
    )  # (Na, M), (Na,)
    zB, yB = collect_latents_and_labels(
        vaeB, dsB, device, args.batch_size, max_samples=args.max_samples
    )  # (Nb, M), (Nb,)

    print(f"Collected zA: {zA.shape}, zB: {zB.shape}")
    print("Label counts A:", torch.bincount(yA, minlength=10))
    print("Label counts B:", torch.bincount(yB, minlength=10))

    Na, M = zA.shape
    Nb, _ = zB.shape

    # Move to device for OT
    zA_dev = zA.to(device)
    zB_dev = zB.to(device)

    # --- Cost matrix: squared Euclidean distances between zA and zB ---
    x_norm = (zA_dev ** 2).sum(dim=1, keepdim=True)  # (Na, 1)
    y_norm = (zB_dev ** 2).sum(dim=1, keepdim=True).t()  # (1, Nb)
    C = x_norm + y_norm - 2.0 * (zA_dev @ zB_dev.t())    # (Na, Nb)
    C = torch.clamp(C, min=0.0)

    # --- Compute entropic OT plan ---
    print("Running Sinkhorn OT...")
    Pi = sinkhorn_ot(
        cost=C,
        epsilon=args.epsilon,
        max_iters=args.sinkhorn_iters,
        tol=args.sinkhorn_tol,
    )  # (Na, Nb)

    # --- Barycentric projections: A->B and B->A ---
    with torch.no_grad():
        # A -> B
        row_sums = Pi.sum(dim=1, keepdim=True)  # (Na, 1)
        row_sums = torch.clamp(row_sums, min=1e-12)
        zA_to_B = Pi @ zB_dev  # (Na, Nb) @ (Nb, M) -> (Na, M)
        zA_to_B = zA_to_B / row_sums

        # B -> A
        col_sums = Pi.sum(dim=0, keepdim=True)  # (1, Nb)
        col_sums = torch.clamp(col_sums, min=1e-12)
        zB_to_A = Pi.t() @ zA_dev  # (Nb, Na) @ (Na, M) -> (Nb, M)
        zB_to_A = zB_to_A / col_sums.t()  # (Nb, 1)

    # Move projections and Pi back to CPU
    zA_to_B = zA_to_B.cpu()
    zB_to_A = zB_to_A.cpu()
    Pi_cpu = Pi.cpu()

    # --- Save everything needed for later evaluation ---
    torch.save(
        {
            "zA": zA,                # (Na, M)
            "zB": zB,                # (Nb, M)
            "yA": yA,                # (Na,)
            "yB": yB,                # (Nb,)
            "Pi": Pi_cpu,            # (Na, Nb)
            "zA_to_B": zA_to_B,      # (Na, M)
            "zB_to_A": zB_to_A,      # (Nb, M)
            "epsilon": args.epsilon,
        },
        args.out_file,
    )
    print(f"Saved OT bridge to {args.out_file}")


if __name__ == "__main__":
    main()
