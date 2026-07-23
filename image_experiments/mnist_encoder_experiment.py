#%%
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)  # avoids severe thread-contention slowdowns on this machine

from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import load_mnist_digits

DIGITS = (6, 8)
MISLABEL_FRAC = 0.2
SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

COLORS = {DIGITS[0]: "#1f77b4", DIGITS[1]: "#d62728"}


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


#%%
torch.manual_seed(SEED)
data = load_mnist_digits(digits=DIGITS, mislabel_frac=MISLABEL_FRAC, seed=SEED)
train, test = data["train"], data["test"]

train_loader = DataLoader(
    TensorDataset(train["images"], train["targets"]),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = Encoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

#%%
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for images, targets in train_loader:
        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.shape[0]
    epoch_loss /= len(train_loader.dataset)

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test["images"]), test["targets"]).item()
        print(f"epoch {epoch:3d}  train_mse={epoch_loss:.4f}  test_mse={test_loss:.4f}")

#%%
model.eval()
with torch.no_grad():
    train_encoded = model(train["images"])
    test_encoded = model(test["images"])


def plot_pca(encoded, targets, true_labels, is_mislabeled, title, ax):
    pca = PCA(n_components=2)
    stacked = torch.cat([encoded, targets], dim=0).numpy()
    pca.fit(stacked)

    encoded_2d = pca.transform(encoded.numpy())
    targets_2d = pca.transform(targets.numpy())

    true_labels = true_labels.numpy()
    is_mislabeled = is_mislabeled.numpy()

    for digit, color in COLORS.items():
        digit_mask = true_labels == digit

        clean_mask = digit_mask & ~is_mislabeled
        ax.scatter(
            encoded_2d[clean_mask, 0], encoded_2d[clean_mask, 1],
            s=10, color=color, alpha=0.6, label=f"encoder output, digit {digit}",
        )

        mislabeled_mask = digit_mask & is_mislabeled
        if mislabeled_mask.any():
            ax.scatter(
                encoded_2d[mislabeled_mask, 0], encoded_2d[mislabeled_mask, 1],
                s=25, facecolors="none", edgecolors=color, linewidths=1.2,
                label=f"encoder output, digit {digit} (mislabeled)",
            )

        target_digit_2d = targets_2d[digit_mask]
        ax.scatter(
            target_digit_2d[:, 0], target_digit_2d[:, 1],
            marker="*", s=250, color=color, edgecolors="black", linewidths=0.5,
            label=f"one-hot target, digit {digit}",
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_pca(
    train_encoded, train["targets"], train["true_labels"], train["is_mislabeled"],
    "Train: encoder output vs. one-hot target (PCA)", axes[0],
)
plot_pca(
    test_encoded, test["targets"], test["true_labels"], test["is_mislabeled"],
    "Test: encoder output vs. one-hot target (PCA)", axes[1],
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig("pca_plots.png", dpi=150, bbox_inches="tight")
plt.show()
