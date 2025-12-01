import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.supervised_vae import SupervisedVAE
from torch.utils.data import TensorDataset



# --------------------------------------------------------
# Translator model (old MLP – kept for reference)
# --------------------------------------------------------
class TranslatorMLP(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -------------------------------------------------------
# Affine coupling layer (RealNVP)
# -------------------------------------------------------
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden=128, mask=None):
        super().__init__()
        self.dim = dim
        # mask is registered as a buffer so it moves with .to(device)
        self.register_buffer("mask", mask)

        # scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Tanh(),  # keeps scale stable
        )

        self.trans_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        # x: (batch, dim)
        x_id = x * self.mask           # part that stays
        x_change = x * (1.0 - self.mask)

        s = self.scale_net(x_id)
        t = self.trans_net(x_id)

        y_change = x_change * torch.exp(s) + t
        y = x_id + y_change * (1.0 - self.mask)
        return y

    def inverse(self, y):
        y_id = y * self.mask
        y_change = y * (1.0 - self.mask)

        s = self.scale_net(y_id)
        t = self.trans_net(y_id)

        x_change = (y_change - t) * torch.exp(-s)
        x = y_id + x_change * (1.0 - self.mask)
        return x


# -------------------------------------------------------
# Full Translator Flow (stack of coupling layers) – bijective
# -------------------------------------------------------
class TranslatorFlow(nn.Module):
    def __init__(self, latent_dim, num_layers=6):
        super().__init__()
        self.dim = latent_dim
        self.layers = nn.ModuleList()

        masks = []
        for i in range(num_layers):
            if i % 2 == 0:
                mask = self._make_mask(latent_dim, left=True)
            else:
                mask = self._make_mask(latent_dim, left=False)
            masks.append(mask)

        for mask in masks:
            self.layers.append(CouplingLayer(latent_dim, hidden=128, mask=mask))

    def _make_mask(self, dim, left=True):
        mask = torch.zeros(dim, dtype=torch.float32)
        if left:
            mask[: dim // 2] = 1.0
        else:
            mask[dim // 2 :] = 1.0
        return mask

    def forward(self, z):
        # A -> B
        for layer in self.layers:
            z = layer(z)
        return z

    def inverse(self, z):
        # B -> A
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


# --------------------------------------------------------
# Discriminator model (latent-space adversary)
# --------------------------------------------------------
class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --------------------------------------------------------
# Geometry utilities
# --------------------------------------------------------
def pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(1) - z.unsqueeze(0)  # (n, n, d)
    return torch.norm(diff, dim=-1)        # (n, n)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1)
    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
    gamma = 1.0 / (2 * sigma * sigma)
    return (
        torch.exp(-gamma * xx).mean()
        + torch.exp(-gamma * yy).mean()
        - 2 * torch.exp(-gamma * xy).mean()
    )


# --------------------------------------------------------
# Data loaders
# --------------------------------------------------------
def make_domain_loaders(data_root: str, split_path: str, batch_size: int):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        data_root,
        train=True,
        download=True,
        transform=transform,
    )

    # Restrict to digits 1,2,3
    mask = (full_train.targets == 1) | (full_train.targets == 2) | (full_train.targets == 3)
    filtered_indices = mask.nonzero(as_tuple=False).view(-1)
    full_train = Subset(full_train, filtered_indices)

    # Split indices are relative to this filtered subset
    split = torch.load(split_path)
    idxA = split["idxA"]
    idxB = split["idxB"]

    dsA = Subset(full_train, idxA)
    dsB = Subset(full_train, idxB)

    loaderA = DataLoader(
        dsA, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    loaderB = DataLoader(
        dsB, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    return loaderA, loaderB

def make_latent_loaders(latentsA_path: str,
                        latentsB_path: str,
                        batch_size: int):
    """
    Build DataLoaders over *latent* datasets saved as:
      {"z": (N, d), "y": (N,)}
    """
    dataA = torch.load(latentsA_path)
    dataB = torch.load(latentsB_path)

    zA, yA = dataA["z"], dataA["y"]
    zB, yB = dataB["z"], dataB["y"]

    dsA = TensorDataset(zA, yA)
    dsB = TensorDataset(zB, yB)

    loaderA = DataLoader(dsA, batch_size=batch_size,
                         shuffle=True, num_workers=0, pin_memory=False)
    loaderB = DataLoader(dsB, batch_size=batch_size,
                         shuffle=True, num_workers=0, pin_memory=False)
    return loaderA, loaderB


# --------------------------------------------------------
# Main training function
# --------------------------------------------------------
def train_translators(args):
    device = args.device

    # Load VAEs (frozen)
    vaeA = SupervisedVAE(latent_dim=args.latent_dim)
    # vaeB = SupervisedVAE(latent_dim=args.latent_dim)
    vaeA.load_state_dict(torch.load(args.weights_vaeA, map_location=device))
    # vaeB.load_state_dict(torch.load(args.weights_vaeB, map_location=device))

    vaeA.to(device).eval()
    for p in vaeA.parameters():
        p.requires_grad = False

    # Translator: single bijective flow
    T_AB = TranslatorFlow(args.latent_dim).to(device)

    # Discriminators (latent-domain)
    D_B_disc = LatentDiscriminator(args.latent_dim).to(device)
    D_A_disc = LatentDiscriminator(args.latent_dim).to(device)

    # ------ semantic anchors: fixed structure in A/B latent spaces ------
    anchors = torch.load(args.sem_anchors_path, map_location=device)
    muA = anchors["muA"].to(device)               # (K, latent_dim)
    muB = anchors["muB_aligned"].to(device)       # (K, latent_dim)
    # digits = anchors["digits"]  # not used in the loss, but good for debugging

    # Optimizers
    opt_T = optim.Adam(
        T_AB.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    opt_D = optim.Adam(
        list(D_B_disc.parameters()) + list(D_A_disc.parameters()),
        lr=args.lr_disc,
    )

    bce = nn.BCELoss()

    # Latent loaders: only zA/zB and labels, no encoders
    loaderA, loaderB = make_latent_loaders(
        args.latentsA_path, args.latentsB_path, args.batch_size
    )
    iterA = iter(loaderA)
    iterB = iter(loaderB)

    for step in range(1, args.steps + 1):
        # ---------------------------
        # Load latent batch (A, B)
        # ---------------------------
        try:
            zA, yA = next(iterA)
        except StopIteration:
            iterA = iter(loaderA)
            zA, yA = next(iterA)

        try:
            zB, yB = next(iterB)
        except StopIteration:
            iterB = iter(loaderB)
            zB, yB = next(iterB)

        zA = zA.to(device)
        zB = zB.to(device)
        yA = yA.to(device)
        yB = yB.to(device)

        # ---------------------------
        # Forward translations (bijective)
        # ---------------------------
        zA_B = T_AB(zA)          # A -> B
        zB_A = T_AB.inverse(zB)  # B -> A

        # ----------------------------------------------------
        # Semantic loss – anchor based
        # ----------------------------------------------------
        # Map B’s cluster centers into A’s latent space
        muB_in_A = T_AB.inverse(muB)          # (K, latent_dim)

        # Force each mapped B-cluster center to match A’s class mean
        sem_loss = (muB_in_A - muA).pow(2).mean()


        # ---------------------------
        # Cycle losses (Removed again)
        # ---------------------------
        zA_cycle = T_AB.inverse(zA_B)
        zB_cycle = T_AB(zB_A)

        cycle_loss = (zA_cycle - zA).pow(2).mean() + (zB_cycle - zB).pow(2).mean()

        # ---------------------------
        # Geometry losses
        # ---------------------------
        DA = pairwise_distances(zA)
        DA_m = pairwise_distances(zA_B)
        DB = pairwise_distances(zB)
        DB_m = pairwise_distances(zB_A)

        geom_loss = (DA - DA_m).pow(2).mean() + (DB - DB_m).pow(2).mean()

        # ---------------------------
        # MMD loss (optional)
        # ---------------------------
        if args.w_mmd > 0:
            mmd_loss = mmd_rbf(zA_B, zB, args.mmd_sigma) + mmd_rbf(
                zB_A, zA, args.mmd_sigma
            )
        else:
            mmd_loss = torch.tensor(0.0, device=device)

        # ---------------------------
        # --- Adversarial losses ---
        # ---------------------------
        if args.w_adv > 0:
            # Train discriminators
            opt_D.zero_grad()

            # B-side discriminator: real B vs translated A->B
            real_B = D_B_disc(zB)
            fake_B = D_B_disc(zA_B.detach())
            ones_real_B = torch.ones_like(real_B)
            zeros_fake_B = torch.zeros_like(fake_B)

            loss_D_B = bce(real_B, ones_real_B) + bce(fake_B, zeros_fake_B)

            # A-side discriminator: real A vs translated B->A
            real_A = D_A_disc(zA)
            fake_A = D_A_disc(zB_A.detach())
            ones_real_A = torch.ones_like(real_A)
            zeros_fake_A = torch.zeros_like(fake_A)

            loss_D_A = bce(real_A, ones_real_A) + bce(fake_A, zeros_fake_A)

            loss_D = 0.5 * (loss_D_A + loss_D_B)
            loss_D.backward()
            opt_D.step()

            # Train translator to fool discriminators
            fake_B_for_T = D_B_disc(zA_B)
            fake_A_for_T = D_A_disc(zB_A)

            ones_B_T = torch.ones_like(fake_B_for_T)
            ones_A_T = torch.ones_like(fake_A_for_T)

            adv_loss = 0.5 * (
                bce(fake_B_for_T, ones_B_T) + bce(fake_A_for_T, ones_A_T)
            )
        else:
            adv_loss = torch.tensor(0.0, device=device)

        # # ---------------------------
        # # Semantic loss (A labels only)
        # # ---------------------------
        # # 1) Compute class means in A's latent space for this batch
        # unique_labels = torch.unique(yA)
        # class_means = []
        # for lbl in unique_labels:
        #     mask = (yA == lbl)
        #     class_means.append(zA[mask].mean(dim=0))
        # mu_A = torch.stack(class_means, dim=0)  # (C, dim)

        # # Map each A-sample to its class mean (tighten A clusters)
        # label_to_index = {int(lbl.item()): i for i, lbl in enumerate(unique_labels)}
        # idx = torch.tensor(
        #     [label_to_index[int(lbl.item())] for lbl in yA],
        #     device=device,
        #     dtype=torch.long,
        # )
        # mu_for_A = mu_A[idx]                     # (N_A, dim)
        # sem_A = (zA - mu_for_A).pow(2).sum(dim=1).mean()

        # # 2) For translated B->A latents, pull each point towards its
        # #    nearest A-class mean (not using B labels)
        # diff_B = zB_A.unsqueeze(1) - mu_A.unsqueeze(0)  # (N_B, C, dim)
        # dist2_B = diff_B.pow(2).sum(dim=2)              # (N_B, C)
        # min_dist2_B, _ = dist2_B.min(dim=1)             # (N_B,)
        # sem_B = min_dist2_B.mean()

        # sem_loss = sem_A + sem_B


        # ---------------------------
        # Total translator loss
        # ---------------------------
        loss_T = (
            args.w_cycle * cycle_loss
            + args.w_geom * geom_loss
            + args.w_mmd * mmd_loss
            + args.w_adv * adv_loss
            + args.w_sem * sem_loss
        )

        opt_T.zero_grad()
        loss_T.backward()
        opt_T.step()

        if step % args.log_every == 0:
            log = (
                f"[Step {step:5d}] "
                f"Tot={loss_T.item():.4f} "
                # f"Cyc={cycle_loss.item():.4f} "
                f"Geo={geom_loss.item():.4f} "
                f"Sem={sem_loss.item():.4f} "
            )
            if args.w_mmd > 0:
                log += f"MMD={mmd_loss.item():.4f} "
            log += f"Adv={adv_loss.item():.4f}"
            print(log)

    torch.save(T_AB.state_dict(), f"checkpoints/{args.out_prefix}_T_AB.pt")
    torch.save(D_A_disc.state_dict(), f"checkpoints/{args.out_prefix}_D_A.pt")
    torch.save(D_B_disc.state_dict(), f"checkpoints/{args.out_prefix}_D_B.pt")
    print("Saved translator (flow) + discriminator models.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--weights-vaeA", type=str, default="checkpoints/mnist_split_vaeA.pt")
    ap.add_argument("--weights-vaeB", type=str, default="checkpoints/mnist_split_vaeB.pt")
    ap.add_argument("--split-path", type=str, default="checkpoints/mnist_split_split_indices.pt")
    ap.add_argument("--latentsA-path", type=str, default="checkpoints/mnist_split_zA_train.pt")
    ap.add_argument("--latentsB-path", type=str, default="checkpoints/mnist_split_zB_train.pt")
    ap.add_argument("--sem-anchors-path", type=str, default="checkpoints/mnist_split_sem_anchors.pt")
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-disc", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--w-cycle", type=float, default=0.0)
    ap.add_argument("--w-geom", type=float, default=0.7)
    ap.add_argument("--w-mmd", type=float, default=0.0)
    ap.add_argument("--w-adv", type=float, default=0.3)
    ap.add_argument("--w-sem", type=float, default=0.1)
    ap.add_argument("--mmd-sigma", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-prefix", type=str, default="mnist_split")
    args = ap.parse_args()

    train_translators(args)


if __name__ == "__main__":
    main()
