import itertools
import torch
from sklearn.cluster import KMeans

from utils.compute_A_stats import compute_A_stats


def compute_cluster_stats(z, cluster_labels, num_clusters, dist_edges, num_pairs=20000):
    """
    z: (N,D) tensor (på CPU)
    cluster_labels: (N,) long tensor med værdier i {0,...,K-1}
    dist_edges: (B+1,) tensor fra A-stats
    return:
      means: (K,D)
      covs:  (K,D,D)
      hists: (K,B)
    """
    device = z.device
    K = num_clusters
    N, D = z.shape
    B = dist_edges.numel() - 1

    means = torch.empty(K, D, device=device)
    covs = torch.empty(K, D, D, device=device)
    hists = torch.empty(K, B, device=device)

    for k in range(K):
        idx = (cluster_labels == k).nonzero(as_tuple=False).view(-1)
        zk = z[idx]
        n = zk.shape[0]
        if n < 2:
            # fallback hvis et cluster skulle være næsten tomt
            means[k] = zk.mean(dim=0) if n > 0 else torch.zeros(D, device=device)
            covs[k] = torch.eye(D, device=device)
            hists[k] = torch.zeros(B, device=device)
            continue

        # mean + cov
        mu = zk.mean(dim=0)
        means[k] = mu
        zk_centered = zk - mu
        covs[k] = zk_centered.T @ zk_centered / (n - 1)

        # afstands-histogram
        m = min(num_pairs, n * n)
        i = torch.randint(0, n, (m,), device=device)
        j = torch.randint(0, n, (m,), device=device)
        d = (zk[i] - zk[j]).norm(dim=1)

        bin_ids = torch.bucketize(d, dist_edges) - 1
        bin_ids = bin_ids.clamp(0, B - 1)
        hist = torch.bincount(bin_ids, minlength=B).float()
        hists[k] = hist

    return means.cpu(), covs.cpu(), hists.cpu()


def build_cost_matrix(mu_A, cov_A, hists_A, mu_B, cov_B, hists_B,
                      alpha=1.0, beta=0.1, gamma=1.0):
    """
    mu_A:    (K,D)
    cov_A:   (K,D,D)
    hists_A: (K,B)
    mu_B:    (K,D)
    cov_B:   (K,D,D)
    hists_B: (K,B)
    return: C (K,K)
    """
    K, D = mu_A.shape
    B = hists_A.shape[1]
    C = torch.empty(K, K)

    # normalisér hist for at få fordelinger
    hA = hists_A / (hists_A.sum(dim=1, keepdim=True) + 1e-8)
    hB = hists_B / (hists_B.sum(dim=1, keepdim=True) + 1e-8)

    for i in range(K):
        for j in range(K):
            mean_dist = torch.norm(mu_A[i] - mu_B[j])
            cov_dist = torch.norm(cov_A[i] - cov_B[j])  # Frobenius
            hist_dist = torch.norm(hA[i] - hB[j])
            C[i, j] = alpha * mean_dist + beta * cov_dist + gamma * hist_dist
    return C


def find_best_permutation(C):
    """
    Brute-force over alle permutationer (K=3 → 6 permutationer).
    C: (K,K) cost-matrix
    return: best_perm (tuple), best_cost (float)
    """
    K = C.shape[0]
    best_perm = None
    best_cost = None
    for perm in itertools.permutations(range(K)):
        cost = sum(C[i, perm[i]].item() for i in range(K))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    return best_perm, best_cost


def evaluate_mapping(cluster_labels, yB, classes_A, best_perm):
    """
    cluster_labels: (N_B,) i {0,1,2}
    yB:             (N_B,) sande digits (1,2,3)
    classes_A:      (K,) tensor med A-digits, fx [1,2,3]
    best_perm:      tuple længde K, hvor best_perm[i] = cluster-index for A-klasse i
    """
    K = len(best_perm)
    N = yB.shape[0]

    # cluster j -> hvilken digit forudsiger vi (fra A)
    cluster_to_digit = {}
    for i in range(K):
        digit_i = int(classes_A[i].item())
        cluster_j = best_perm[i]
        cluster_to_digit[cluster_j] = digit_i

    # cluster-purity og global accuracy
    hits = 0
    total = N

    for j in range(K):
        idx = (cluster_labels == j).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        y_cluster = yB[idx]
        # majority-label i cluster (til analyse)
        values, counts = y_cluster.unique(return_counts=True)
        majority_label = int(values[counts.argmax()].item())

        predicted_digit = cluster_to_digit[j]
        correct = (y_cluster == predicted_digit).sum().item()
        cluster_acc = correct / idx.numel()

        hits += correct

        print(f"Cluster {j}: size={idx.numel()}, "
              f"majority_label={majority_label}, "
              f"pred_digit={predicted_digit}, "
              f"cluster_acc={cluster_acc:.3f}")

    overall_acc = hits / total
    print(f"Overall B-side accuracy given mapping A→clusters: {overall_acc:.3f}")


def main():
    prefix = "checkpoints/mnist_split"

    # 1) A-stats
    # compute_A_stats(prefix=prefix)  

    # 2) Load A-stats
    statsA = torch.load(f"{prefix}_zA_stats.pt")
    classes_A = statsA["classes"]              # (3,)
    mu_A = statsA["class_means"]               # (3,D)
    cov_A = statsA["class_covs"]               # (3,D,D)
    dist_edges = statsA["dist_hist_edges"]     # (B+1,)
    hists_A = statsA["class_dist_hists"]       # (3,B)

    # 3) Load B-latents (rå)
    dataB = torch.load(f"{prefix}_zB_train.pt")
    zB = dataB["z"]    # (N_B,D), antages at være på CPU
    yB = dataB["y"]    # (N_B,)

    # 4) K-means på zB (unsupervised clustre)
    K = classes_A.numel()
    km = KMeans(n_clusters=K, n_init=10)
    cluster_labels_np = km.fit_predict(zB.numpy())
    cluster_labels = torch.from_numpy(cluster_labels_np).long()

    # 5) Cluster-stats for B-clustre (samme bins som A)
    mu_B, cov_B, hists_B = compute_cluster_stats(
        zB, cluster_labels, num_clusters=K, dist_edges=dist_edges
    )

    # 6) 3x3 cost-matrix
    C = build_cost_matrix(mu_A, cov_A, hists_A, mu_B, cov_B, hists_B)
    print("Cost matrix C (rows=A-classes, cols=B-clusters):")
    print(C)

    # 7) Bedste permutation (A-klasse i -> cluster perm[i])
    best_perm, best_cost = find_best_permutation(C)
    print(f"Best permutation (A-class index -> B-cluster): {best_perm}, cost={best_cost:.4f}")

    # 8) Evaluer mod sande labels yB
    evaluate_mapping(cluster_labels, yB, classes_A, best_perm)


if __name__ == "__main__":
    main()
