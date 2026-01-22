import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden=128):
        super().__init__()
        self.register_buffer("mask", mask)  # (dim,)
        self.s_net = MLP(dim, dim, hidden)
        self.t_net = MLP(dim, dim, hidden)

    def forward(self, x):
        # x -> y, logdet
        x_masked = x * self.mask
        s = self.s_net(x_masked) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)

        s = torch.tanh(s)  # stabilitet
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = (s).sum(dim=1)
        return y, logdet

    def inverse(self, y):
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)

        s = torch.tanh(s)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class RealNVP(nn.Module):
    def __init__(self, dim, n_layers=6, hidden=128):
        super().__init__()
        masks = []
        for k in range(n_layers):
            m = torch.zeros(dim)
            m[k % 2::2] = 1.0  # skiftevis maske
            masks.append(m)
        self.layers = nn.ModuleList([AffineCoupling(dim, mask=masks[k], hidden=hidden) for k in range(n_layers)])

    def forward(self, x):
        logdet_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, logdet = layer(z)
            logdet_sum += logdet
        return z, logdet_sum

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


def main():
    prefix = "checkpoints/mnist_split"
    device = "cpu"

    data = torch.load(f"{prefix}_gw_pairs.pt")
    zA = data["zA"].float().to(device)
    zB_t = data["zB_target"].float().to(device)

    D = zA.shape[1]
    flow = RealNVP(dim=D, n_layers=8, hidden=128).to(device)

    opt = optim.Adam(flow.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 256
    n = zA.shape[0]
    idx = torch.arange(n, device=device)

    for epoch in range(1, 1001):
        perm = idx[torch.randperm(n)]
        total = 0.0
        flow.train()

        for s in range(0, n, batch_size):
            b = perm[s:s+batch_size]
            x = zA[b]
            y = zB_t[b]

            z_pred, _ = flow(x)
            loss = loss_fn(z_pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]

        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  mse={total/n:.6f}")

    torch.save(flow.state_dict(), f"{prefix}_flowT_gw.pt")
    print(f"Saved flow translator to {prefix}_flowT_gw.pt")


if __name__ == "__main__":
    main()
