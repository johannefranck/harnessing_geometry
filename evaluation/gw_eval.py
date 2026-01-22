import numpy as np
import torch
import ot  # POT

def pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    # x: (n,d)
    # return: (n,n) squared euclidean distances
    xx = np.sum(x * x, axis=1, keepdims=True)
    D = xx + xx.T - 2.0 * (x @ x.T)
    D[D < 0] = 0
    return D

def barycentric_map(Pi: np.ndarray, XB: np.ndarray) -> np.ndarray:
    # Pi: (nA,nB) coupling
    # XB: (nB,d)
    # returns mapped XA->B barycenters: (nA,d)
    row_sums = Pi.sum(axis=1, keepdims=True) + 1e-12
    W = Pi / row_sums
    return W @ XB

def nn_accuracy(mapped: np.ndarray, XB: np.ndarray, yB: np.ndarray) -> float:
    # nearest neighbor in B for each mapped point
    # mapped: (nA,d), XB: (nB,d), yB: (nB,)
    # returns acc vs yA that we pass separately outside
    # compute distances in blocks to avoid huge memory if needed
    dists = ((mapped[:, None, :] - XB[None, :, :]) ** 2).sum(axis=2)  # (nA,nB)
    nn = dists.argmin(axis=1)
    return nn, yB[nn]

def main():
    prefix = "checkpoints/mnist_split"
    n = 800              # subsample size per domain (start 500–1000)
    reg = 0.1           # entropic regularization for GW
    max_iter = 200       # GW iterations

    # Load latents
    A = torch.load(f"{prefix}_zA_train.pt")
    B = torch.load(f"{prefix}_zB_train.pt")
    zA = A["z"].cpu()
    yA = A["y"].cpu()
    zB = B["z"].cpu()
    yB = B["y"].cpu()  # only for evaluation

    # Restrict to digits 1,2,3 (should already be true, but safe)
    maskA = (yA == 1) | (yA == 2) | (yA == 3)
    maskB = (yB == 1) | (yB == 2) | (yB == 3)
    zA, yA = zA[maskA], yA[maskA]
    zB, yB = zB[maskB], yB[maskB]

    # Subsample
    g = torch.Generator().manual_seed(0)
    idxA = torch.randperm(zA.shape[0], generator=g)[:n]
    idxB = torch.randperm(zB.shape[0], generator=g)[:n]
    XA = zA[idxA].numpy().astype(np.float64)
    XB = zB[idxB].numpy().astype(np.float64)
    yA_s = yA[idxA].numpy()
    yB_s = yB[idxB].numpy()

    # Intra-domain distance matrices
    DA = pairwise_sq_dists(XA)
    DB = pairwise_sq_dists(XB)

    #stabilize and normalize
    DA = np.sqrt(DA + 1e-12)
    DB = np.sqrt(DB + 1e-12)

    DA = DA / (DA.mean() + 1e-12)
    DB = DB / (DB.mean() + 1e-12)

    # Uniform weights
    p = np.ones((n,), dtype=np.float64) / n
    q = np.ones((n,), dtype=np.float64) / n

    # GW coupling (entropic)
    # returns Pi shape (n,n)
    Pi = ot.gromov.entropic_gromov_wasserstein(
    DA, DB, p, q,
    loss_fun="square_loss",
    epsilon=reg,
    max_iter=max_iter,
    verbose=True
)

    # Barycentric map A->B in latent space
    Xmap = barycentric_map(Pi, XB)  # (n,d)

    # Evaluate via NN in B (subsampled) using yB_s
    nn_idx, pred_y = nn_accuracy(Xmap, XB, yB_s)
    acc = (pred_y == yA_s).mean()

    print(f"\nGW barycentric NN accuracy on subsample (n={n}): {acc:.3f}")

    # Also print per-class accuracy
    for digit in [1, 2, 3]:
        m = (yA_s == digit)
        if m.sum() > 0:
            a = (pred_y[m] == yA_s[m]).mean()
            print(f"  digit {digit}: acc={a:.3f} (count={m.sum()})")

    # Save pseudo-pairs for next step (translator training)
    # pairs: (XA, Xmap) meaning zA -> mapped zB target
    out = {
        "zA": torch.from_numpy(XA).float(),
        "yA": torch.from_numpy(yA_s).long(),
        "zB_target": torch.from_numpy(Xmap).float(),
        "Pi": torch.from_numpy(Pi).float(),
        "idxA": idxA,
        "idxB": idxB,
    }
    torch.save(out, f"{prefix}_gw_pairs.pt")
    print(f"Saved GW pseudo-pairs to {prefix}_gw_pairs.pt")

if __name__ == "__main__":
    main()
