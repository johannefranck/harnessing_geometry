import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.supervised_vae import SupervisedVAE
from training.train_translators_planA_adv import TranslatorMLP, TranslatorFlow


def collect_latents_with_labels(vae,
                                loader: DataLoader,
                                device: str,
                                max_samples: int):
    """
    Encode up to max_samples images into latent means + labels.
    Returns:
        z:      (N, latent_dim) tensor of latents
        y_all:  (N,) tensor of labels
    """
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
    return z, y_all


class LatentClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor):
        return self.linear(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--weights-vaeA", type=str, default="checkpoints/mnist_split_vaeA.pt")
    ap.add_argument("--weights-vaeB", type=str, default="checkpoints/mnist_split_vaeB.pt")
    ap.add_argument("--weights-TBA", type=str, default="checkpoints/mnist_split_T_AB.pt")
    ap.add_argument("--split-path", type=str, default="checkpoints/mnist_split_split_indices.pt")
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-train-samples", type=int, default=10000)
    ap.add_argument("--max-test-samples", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = args.device

    # --- Load VAEs (SupervisedVAE, matching training) ---
    vaeA = SupervisedVAE(latent_dim=args.latent_dim)
    vaeA.load_state_dict(torch.load(args.weights_vaeA, map_location=device))
    vaeB = SupervisedVAE(latent_dim=args.latent_dim)
    vaeB.load_state_dict(torch.load(args.weights_vaeB, map_location=device))
    vaeA.to(device)
    vaeB.to(device)

    # --- Load TranslatorFlow (we will use its inverse as B->A) ---
    T_BA = TranslatorFlow(args.latent_dim)
    T_BA.load_state_dict(torch.load(args.weights_TBA, map_location=device))
    T_BA.to(device).eval()

    # --- Load MNIST train set and split indices (for train-A only) ---
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        args.data_root, train=True, download=True, transform=transform
    )

    # Restrict to digits 1,2,3, consistent with your VAE training
    mask = (full_train.targets == 1) | (full_train.targets == 2) | (full_train.targets == 3)
    filtered_indices = mask.nonzero(as_tuple=False).view(-1)
    full_train = Subset(full_train, filtered_indices)

    split = torch.load(args.split_path)
    idxA = split["idxA"]   # indices for domain A (relative to filtered subset)
    dsA_train = Subset(full_train, idxA)

    train_loader_A = DataLoader(
        dsA_train, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )

    # --- Collect latents for train-A (for classifier training) ---
    zA_train, y_train = collect_latents_with_labels(
        vaeA, train_loader_A, device, max_samples=args.max_train_samples
    )
    print(f"Collected zA_train: {zA_train.shape}")

    # ---- prepare the classifier to understand the digits correctly ---
    y_train = y_train - 1  # {1,2,3} -> {0,1,2}

    # --- Define classifier in A-space ---
    clf = LatentClassifier(latent_dim=args.latent_dim, num_classes=3).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # --- Train classifier on A latents only ---
    clf.train()
    dataset_size = zA_train.size(0)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(dataset_size)
        zA_shuf = zA_train[perm]
        y_shuf = y_train[perm]

        total_loss = 0.0
        correct = 0
        count = 0

        for start in range(0, dataset_size, args.batch_size):
            end = start + args.batch_size
            z_batch = zA_shuf[start:end].to(device)
            y_batch = y_shuf[start:end].to(device)

            optimizer.zero_grad()
            logits = clf(z_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * z_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            count += z_batch.size(0)

        avg_loss = total_loss / count
        acc = correct / count
        print(f"[Epoch {epoch:2d}] train loss={avg_loss:.4f}, train acc={acc:.4f}")

    # --- Prepare TEST data (filtered MNIST test) ---
    test_full = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)

    mask = (test_full.targets == 1) | (test_full.targets == 2) | (test_full.targets == 3)
    filtered_idx = mask.nonzero(as_tuple=False).view(-1)

    test_set = Subset(test_full, filtered_idx)

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # Collect latents for TEST in both spaces
    zA_test, y_test = collect_latents_with_labels(
        vaeA, test_loader, device, max_samples=args.max_test_samples
    )
    zB_test, _ = collect_latents_with_labels(
        vaeB, test_loader, device, max_samples=args.max_test_samples
    )

    print(f"Collected zA_test: {zA_test.shape}, zB_test: {zB_test.shape}")
    # remap digits to match classifier training
    y_test = y_test - 1  # {1,2,3} -> {0,1,2}

    # --- Evaluate classifier on A-test (baseline) ---
    clf.eval()
    with torch.no_grad():
        logits_A = clf(zA_test.to(device))
        preds_A = logits_A.argmax(dim=1).cpu()
        acc_A = (preds_A == y_test).float().mean().item()

    print(f"Classifier accuracy on A-test (zA_test): {acc_A:.4f}")

    # --- Evaluate classifier on translated B-test (B -> A via inverse flow) ---
    with torch.no_grad():
        # T_BA here is actually the flow T_AB; we use its inverse as B->A
        zB_to_A = T_BA.inverse(zB_test.to(device))
        logits_BA = clf(zB_to_A)
        preds_BA = logits_BA.argmax(dim=1).cpu()
        acc_BA = (preds_BA == y_test).float().mean().item()

    print(f"Classifier accuracy on translated B-test (T_BA(zB_test) via inverse flow): {acc_BA:.4f}")


if __name__ == "__main__":
    main()
