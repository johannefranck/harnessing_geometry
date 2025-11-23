import argparse
import torch


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


class LatentClassifier(torch.nn.Module):
    def __init__(self, latent_dim: int, num_classes: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor):
        return self.linear(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--ot-file", type=str, default="checkpoints/mnist_ot_planB.pt")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k-nn", type=int, default=10)
    ap.add_argument("--clf-epochs", type=int, default=20)
    ap.add_argument("--clf-lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = args.device

    # --- Load OT bridge (latents + labels) ---
    ot = torch.load(args.ot_file, map_location="cpu")
    zA = ot["zA"]          # (Na, M)
    zB = ot["zB"]          # (Nb, M)
    yA = ot["yA"]          # (Na,)
    yB = ot["yB"]          # (Nb,)
    zA_to_B = ot["zA_to_B"]  # (Na, M)
    zB_to_A = ot["zB_to_A"]  # (Nb, M)

    Na, M = zA.shape
    Nb, M2 = zB.shape
    assert M == M2 == args.latent_dim, "latent-dim mismatch"

    print(f"Loaded OT bridge: zA {zA.shape}, zB {zB.shape}")
    print("Label counts A:", torch.bincount(yA, minlength=10))
    print("Label counts B:", torch.bincount(yB, minlength=10))

    # --- Geometry: distance correlation & kNN overlap ---
    DA = pairwise_distances(zA)
    DA_m = pairwise_distances(zA_to_B)
    DA_flat = upper_triangle_flat(DA)
    DA_m_flat = upper_triangle_flat(DA_m)
    corr_A = pearson_corr_torch(DA_flat, DA_m_flat)
    overlap_A = knn_overlap(zA, zA_to_B, k=args.k_nn)

    DB = pairwise_distances(zB)
    DB_m = pairwise_distances(zB_to_A)
    DB_flat = upper_triangle_flat(DB)
    DB_m_flat = upper_triangle_flat(DB_m)
    corr_B = pearson_corr_torch(DB_flat, DB_m_flat)
    overlap_B = knn_overlap(zB, zB_to_A, k=args.k_nn)

    print(f"Distance correlation A (orig vs mapped): {corr_A:.4f}")
    print(f"Distance correlation B (orig vs mapped): {corr_B:.4f}")
    print(f"{args.k_nn}-NN overlap A (orig vs mapped): {overlap_A:.4f}")
    print(f"{args.k_nn}-NN overlap B (orig vs mapped): {overlap_B:.4f}")

    # --- Class-mean alignment A->B ---
    class_means_A_to_B = []
    class_means_B = []

    for k in range(10):
        maskA = (yA == k)
        maskB = (yB == k)
        if maskA.sum() == 0 or maskB.sum() == 0:
            print(f"Digit {k}: no samples in A or B subset, skipping in mean alignment.")
            class_means_A_to_B.append(None)
            class_means_B.append(None)
            continue

        mu_A_B = zA_to_B[maskA].mean(dim=0)
        mu_B = zB[maskB].mean(dim=0)
        class_means_A_to_B.append(mu_A_B)
        class_means_B.append(mu_B)

    correct = 0
    total = 0
    for k in range(10):
        if class_means_A_to_B[k] is None:
            continue
        mu_A_B = class_means_A_to_B[k]

        dists = []
        for j in range(10):
            if class_means_B[j] is None:
                dists.append(float("inf"))
            else:
                d = torch.norm(mu_A_B - class_means_B[j])
                dists.append(d.item())
        nearest = int(torch.tensor(dists).argmin().item())
        total += 1
        if nearest == k:
            correct += 1
        print(f"Digit {k}: nearest in B-space is {nearest} (dist={dists[nearest]:.4f})")

    acc_means = correct / total if total > 0 else float("nan")
    print(f"Class-mean alignment accuracy (A→B): {correct}/{total} = {acc_means:.2f}")

    # --- Classifier transfer: train on zA, test on zB_to_A ---
    latent_dim = zA.size(1)
    clf = LatentClassifier(latent_dim=latent_dim, num_classes=10).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.clf_lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Train classifier on zA, yA
    clf.train()
    dataset_size = zA.size(0)
    for epoch in range(1, args.clf_epochs + 1):
        perm = torch.randperm(dataset_size)
        zA_shuf = zA[perm]
        yA_shuf = yA[perm]

        total_loss = 0.0
        correct_train = 0
        count = 0

        for start in range(0, dataset_size, args.batch_size):
            end = start + args.batch_size
            z_batch = zA_shuf[start:end].to(device)
            y_batch = yA_shuf[start:end].to(device)

            optimizer.zero_grad()
            logits = clf(z_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * z_batch.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == y_batch).sum().item()
            count += z_batch.size(0)

        avg_loss = total_loss / count
        acc_train = correct_train / count
        print(f"[Epoch {epoch:2d}] train loss={avg_loss:.4f}, train acc={acc_train:.4f}")

    # Evaluate on A (baseline)
    clf.eval()
    with torch.no_grad():
        logits_A = clf(zA.to(device))
        preds_A = logits_A.argmax(dim=1).cpu()
        acc_A = (preds_A == yA).float().mean().item()
    print(f"Classifier accuracy on A-latents (train subset): {acc_A:.4f}")

    # Evaluate on B->A mapped latents
    with torch.no_grad():
        logits_BA = clf(zB_to_A.to(device))
        preds_BA = logits_BA.argmax(dim=1).cpu()
        acc_BA = (preds_BA == yB).float().mean().item()
    print(f"Classifier accuracy on OT-translated B-latents (B→A): {acc_BA:.4f}")


if __name__ == "__main__":
    main()
