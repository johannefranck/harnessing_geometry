import torch
import torch.nn as nn
import torch.optim as optim


def main():
    prefix = "checkpoints/mnist_split"
    data = torch.load(f"{prefix}_gw_pairs.pt")
    zA = data["zA"]          # (n,D)
    zB_t = data["zB_target"] # (n,D)

    D = zA.shape[1]
    model = nn.Linear(D, D, bias=True)

    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 501):
        opt.zero_grad()
        pred = model(zA)
        loss = loss_fn(pred, zB_t)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  mse={loss.item():.6f}")

    torch.save(model.state_dict(), f"{prefix}_linT_gw.pt")
    print(f"Saved translator to {prefix}_linT_gw.pt")


if __name__ == "__main__":
    main()
