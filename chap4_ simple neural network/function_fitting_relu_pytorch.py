import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def target_function(x: np.ndarray) -> np.ndarray:
    """Nonlinear target function to be fitted."""
    return np.sin(2 * x) + 0.3 * (x ** 2) - 0.5 * np.cos(3 * x)


def build_dataset(num_samples: int = 1200):
    x = np.random.uniform(-3.0, 3.0, size=(num_samples, 1)).astype(np.float32)
    noise = np.random.normal(0.0, 0.08, size=(num_samples, 1)).astype(np.float32)
    y = target_function(x).astype(np.float32) + noise
    return x, y


class ReLURegressor(nn.Module):
    """Two-layer ReLU network: input -> hidden(ReLU) -> output."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def try_save_plots(model, x_all, y_all, train_mask, save_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[INFO] matplotlib is unavailable, skip figure generation.")
        return None, None

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        x_grid = np.linspace(-3.0, 3.0, 600, dtype=np.float32).reshape(-1, 1)
        x_grid_t = torch.from_numpy(x_grid)
        y_pred_grid = model(x_grid_t).cpu().numpy()
        y_true_grid = target_function(x_grid)

    fit_path = os.path.join(save_dir, "function_fit_result.png")
    plt.figure(figsize=(8, 5))
    plt.scatter(x_all[train_mask], y_all[train_mask], s=8, alpha=0.35, label="Train samples")
    plt.scatter(x_all[~train_mask], y_all[~train_mask], s=12, alpha=0.55, label="Test samples")
    plt.plot(x_grid, y_true_grid, color="black", linewidth=2.0, label="Target function")
    plt.plot(x_grid, y_pred_grid, color="#d62728", linewidth=2.0, label="Model prediction")
    plt.title("Two-layer ReLU Network for Function Fitting")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fit_path, dpi=150)
    plt.close()

    return fit_path, None


def main():
    set_seed(42)

    x, y = build_dataset(num_samples=1200)

    # 80/20 split for train and test sets.
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_mask = np.zeros_like(indices, dtype=bool)
    train_mask[:split] = True
    reverse_order = np.argsort(indices)
    train_mask = train_mask[reverse_order]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    model = ReLURegressor(hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    epochs = 2500
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(x_test_t), y_test_t).item()
            print(f"epoch={epoch:4d} train_mse={loss.item():.6f} test_mse={test_loss:.6f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(x_train_t)
        test_pred = model(x_test_t)
        train_mse = criterion(train_pred, y_train_t).item()
        test_mse = criterion(test_pred, y_test_t).item()

        ss_res = torch.sum((y_test_t - test_pred) ** 2)
        ss_tot = torch.sum((y_test_t - torch.mean(y_test_t)) ** 2)
        r2 = (1 - ss_res / ss_tot).item()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path, _ = try_save_plots(model, x, y, train_mask, current_dir)

    print("\n===== Final Metrics =====")
    print(f"train_mse={train_mse:.6f}")
    print(f"test_mse={test_mse:.6f}")
    print(f"test_r2={r2:.6f}")
    if fig_path:
        print(f"fit_figure={fig_path}")


if __name__ == "__main__":
    main()
