import torch
import torch.nn as nn
import torch.optim as optim


def main():
    prefix = "checkpoints/mnist_split"

    # Load translator T: zA -> zB
    A = torch.load(f"{prefix}_zA_train.pt")
    B = torch.load(f"{prefix}_zB_train.pt")
    zA, yA = A["z"].float(), A["y"].long()
    zB, yB = B["z"].float(), B["y"].long()

    D = zA.shape[1]

    T = nn.Linear(D, D, bias=True)
    T.load_state_dict(torch.load(f"{prefix}_linT_gw.pt"))
    T.eval()

    # Train simple classifier on B-latents: C_B(zB) -> {1,2,3}
    # Map labels {1,2,3} to {0,1,2} for CrossEntropy
    yB0 = (yB - 1).clamp(0, 2)
    yA0 = (yA - 1).clamp(0, 2)

    C = nn.Sequential(
        nn.Linear(D, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )

    opt = optim.Adam(C.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # quick training loop
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

    # Evaluate translated A in B-space
    with torch.no_grad():
        zB_hat = T(zA)
        predA = C(zB_hat).argmax(dim=1)
        accA = (predA == yA0).float().mean().item()

    print(f"Accuracy on translated A->B latents (eval via B-classifier): {accA:.3f}")


if __name__ == "__main__":
    main()
