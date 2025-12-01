import argparse
import itertools
import torch
from sklearn.cluster import KMeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zA-path", type=str, default="checkpoints/mnist_split_zA_train.pt")
    ap.add_argument("--zB-path", type=str, default="checkpoints/mnist_split_zB_train.pt")
    ap.add_argument("--out-path", type=str, default="checkpoints/mnist_split_sem_anchors.pt")
    args = ap.parse_args()

    # ---------- load A latents (with labels) ----------
    dataA = torch.load(args.zA_path)
    zA = dataA["z"]        # (NA, d)
    yA = dataA["y"]        # (NA,)

    # digits present (e.g. [1,2,3])
    digits = sorted(yA.unique().tolist())
    K = len(digits)
    d = zA.size(1)

    # class means in A, ordered by 'digits'
    muA = []
    for dgt in digits:
        mask = (yA == dgt)
        muA.append(zA[mask].mean(dim=0))
    muA = torch.stack(muA, dim=0)    # (K, d)

    # ---------- load B latents (no labels used) ----------
    dataB = torch.load(args.zB_path)
    zB = dataB["z"]                  # (NB, d)

    # k-means clusters in B
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    labels_B = kmeans.fit_predict(zB.numpy())
    centers_B = torch.from_numpy(kmeans.cluster_centers_).float()   # (K, d)

    # ---------- find best permutation centers_B -> muA ----------
    perms = list(itertools.permutations(range(K)))
    best_cost = None
    best_perm = None

    for perm in perms:
        perm = list(perm)
        centers_perm = centers_B[perm]                 # (K, d)
        cost = ((centers_perm - muA)**2).sum().item()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    best_perm = list(best_perm)
    muB_aligned = centers_B[best_perm]   # cluster k now aligned to digit digits[k]

    torch.save(
        {
            "muA": muA,                     # (K, d), ordered by 'digits'
            "muB_aligned": muB_aligned,     # (K, d), aligned to same order
            "digits": torch.tensor(digits, dtype=torch.long),
        },
        args.out_path,
    )
    print(f"Saved semantic anchors to {args.out_path}")
    print("digits:", digits)
    print("muA shape:", muA.shape, "muB_aligned shape:", muB_aligned.shape)


if __name__ == "__main__":
    main()
