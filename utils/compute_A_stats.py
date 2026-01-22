import torch

def compute_A_stats(prefix="checkpoints/mnist_split", B=50, num_pairs=20000):
    # 1) Load A-latents
    data = torch.load(f"{prefix}_zA_train.pt")
    z = data["z"]     # (N_A, D)
    y = data["y"]     # (N_A,)
    classes = torch.tensor([1,2,3])
    K = classes.numel()
    N, D = z.shape

    # 2) Class indices
    class_indices = { int(c): (y==c).nonzero(as_tuple=False).view(-1)
                      for c in classes }

    # 3) Means and covariances
    class_means = torch.empty(K, D)
    class_covs  = torch.empty(K, D, D)

    for k_idx, c in enumerate(classes):
        idx = class_indices[int(c)]
        zk = z[idx]
        mu = zk.mean(dim=0)
        class_means[k_idx] = mu
        zk_centered = zk - mu
        cov = zk_centered.T @ zk_centered / (zk.shape[0]-1)
        class_covs[k_idx] = cov

    # Global stats
    global_mean = z.mean(dim=0)
    zc = z - global_mean
    global_cov = zc.T @ zc / (N - 1)

    # 4) Distance histograms
    # Sample representative distances for all classes
    all_d = []

    for c in classes:
        idx = class_indices[int(c)]
        zk = z[idx]
        n = zk.shape[0]

        i = torch.randint(0, n, (num_pairs,))
        j = torch.randint(0, n, (num_pairs,))
        d = (zk[i] - zk[j]).norm(dim=1)
        all_d.append(d)

    all_d = torch.cat(all_d)
    dist_min, dist_max = all_d.min().item(), all_d.max().item()
    dist_edges = torch.linspace(dist_min, dist_max, B+1)
    class_hists = torch.empty(K, B)

    for k_idx, c in enumerate(classes):
        idx = class_indices[int(c)]
        zk = z[idx]
        n = zk.shape[0]
        i = torch.randint(0, n, (num_pairs,))
        j = torch.randint(0, n, (num_pairs,))
        d = (zk[i] - zk[j]).norm(dim=1)

        bin_ids = torch.bucketize(d, dist_edges) - 1
        bin_ids = bin_ids.clamp(0, B-1)
        hist = torch.bincount(bin_ids, minlength=B).float()
        class_hists[k_idx] = hist

    # 5) Save everything
    out = {
        "z": z,
        "y": y,
        "classes": classes,
        "class_indices": class_indices,
        "class_means": class_means,
        "class_covs": class_covs,
        "global_mean": global_mean,
        "global_cov": global_cov,
        "dist_hist_edges": dist_edges,
        "class_dist_hists": class_hists,
    }

    torch.save(out, f"{prefix}_zA_stats.pt")
    print("Saved A-stats to", f"{prefix}_zA_stats.pt")

if __name__ == "__main__":
    compute_A_stats()