import torch
import torch.nn as nn
import torch.optim as optim

from training.train_flow_gw import RealNVP 


def main():
    prefix = "checkpoints/mnist_split"
    device = "cpu"

    A = torch.load(f"{prefix}_zA_train.pt")
    B = torch.load(f"{prefix}_zB_train.pt")
    zA, yA = A["z"].float().to(device), A["y"].long().to(device)
    zB, yB = B["z"].float().to(device), B["y"].long().to(device)

    D = zA.shape[1]

    # Load flow
    flow = RealNVP(dim=D, n_layers=8, hidden=128).to(device)
    flow.load_state_dict(torch.load(f"{prefix}_flowT_gw.pt", map_location=device))
    flow.eval()

    # Train B-classifier (eval only)
    yB0 = (yB - 1).clamp(0, 2)
    yA0 = (yA - 1).clamp(0, 2)

    C = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 3)).to(device)
    opt = optim.Adam(C.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        opt.zero_grad()
        logits = C(zB)
        loss = loss_fn(logits, yB0)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            accB = (logits.argmax(dim=1) == yB0).float().mean().item()
            print(f"epoch {epoch:3d}  clf_loss={loss.item():.4f}  clf_acc(B)={accB:.3f}")

    C.eval()

    with torch.no_grad():
        zB_hat, _ = flow(zA)          # forward map A->B
        predA = C(zB_hat).argmax(dim=1)
        accA = (predA == yA0).float().mean().item()

    print(f"Accuracy on A->B latents using FLOW (eval via B-classifier): {accA:.3f}")


if __name__ == "__main__":
    main()
