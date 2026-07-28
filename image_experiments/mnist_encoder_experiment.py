#%%
import os

import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)  # avoids severe thread-contention slowdowns on this machine

import wandb
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import load_mnist_digits

SCRATCH_DIR = "/tmp/claude-1003/-home-cmdunham-LatentSpaceExploration/b0fd6c9e-b2c2-4b5c-aca0-8208883a223e/scratchpad"
EMBEDDINGS_PATH = f"{SCRATCH_DIR}/mnist_6_8_embeddings.pt"

DIGITS = (6, 8)
MISLABEL_FRAC = 0.2
VAL_FRAC = 0.15
SEED = 42
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128

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


def split_train_val(train_data, val_frac, seed):
    n = train_data["images"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    subset = lambda idx: {k: v[idx] for k, v in train_data.items()}
    return subset(train_idx), subset(val_idx)


#%%
torch.manual_seed(SEED)
data = load_mnist_digits(digits=DIGITS, mislabel_frac=MISLABEL_FRAC, seed=SEED)
full_train, test = data["train"], data["test"]
train, val = split_train_val(full_train, VAL_FRAC, SEED)

train_loader = DataLoader(
    TensorDataset(train["images"], train["targets"]),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = Encoder(hidden_dim=HIDDEN_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())

run = wandb.init(
    project="mnist-6-8-encoder",
    config={
        "digits": DIGITS,
        "mislabel_frac": MISLABEL_FRAC,
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": 10,
        "architecture": "mlp_784_h_h//2_10_sigmoid",
        "loss_fn": "MSE",
        "optimizer": "Adam",
        "n_train": train["images"].shape[0],
        "n_val": val["images"].shape[0],
        "n_test": test["images"].shape[0],
        "n_params": n_params,
    },
)
wandb.watch(model, log="all", log_freq=len(train_loader))

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

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(val["images"]), val["targets"]).item()
    wandb.log({"epoch": epoch, "train_mse": epoch_loss, "val_mse": val_loss})

    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d}  train_mse={epoch_loss:.4f}  val_mse={val_loss:.4f}")

# Training is complete -- the test set has not been touched until this point.
#%%
model.eval()
with torch.no_grad():
    test_loss = loss_fn(model(test["images"]), test["targets"]).item()
print(f"final test_mse={test_loss:.4f}")
run.summary["test_mse"] = test_loss

with torch.no_grad():
    train_encoded = model(train["images"])
    test_encoded = model(test["images"])

os.makedirs(SCRATCH_DIR, exist_ok=True)
torch.save(
    {
        "run_id": run.id,
        "digits": DIGITS,
        "train_encoded": train_encoded,
        "train_true_labels": train["true_labels"],
        "train_assigned_labels": train["assigned_labels"],
        "train_is_mislabeled": train["is_mislabeled"],
        "test_encoded": test_encoded,
        "test_true_labels": test["true_labels"],
    },
    EMBEDDINGS_PATH,
)
print(f"saved encoder outputs to {EMBEDDINGS_PATH}")


def classify(encoded, true_labels):
    """Predicts each sample's digit as whichever of the two relevant one-hot
    dims scores higher (the other 8 output dims are never a training target,
    so a full 10-way argmax isn't meaningful here). Returns per-digit stats
    (this is *not* logged to W&B, only used for the report and plot titles)."""
    digit_indices = torch.tensor(DIGITS)
    predicted = digit_indices[encoded[:, digit_indices].argmax(dim=1)]
    correct = predicted == true_labels

    stats = {}
    for digit in DIGITS:
        mask = true_labels == digit
        n = mask.sum().item()
        n_correct = correct[mask].sum().item()
        stats[digit] = {
            "n": n,
            "n_correct": n_correct,
            "n_incorrect": n - n_correct,
            "pct_correct": 100 * n_correct / n,
            "pct_incorrect": 100 * (n - n_correct) / n,
        }
    return predicted, correct, stats


train_predicted, train_correct, train_stats = classify(train_encoded, train["true_labels"])
test_predicted, test_correct, test_stats = classify(test_encoded, test["true_labels"])

for split_name, correct, stats in [("train", train_correct, train_stats), ("test", test_correct, test_stats)]:
    print(f"\n{split_name} set classification (encoder output argmax over the 2 relevant dims):")
    for digit in DIGITS:
        s = stats[digit]
        print(f"  digit {digit}: {s['n_correct']}/{s['n']} correct ({s['pct_correct']:.2f}%), "
              f"{s['n_incorrect']}/{s['n']} incorrect ({s['pct_incorrect']:.2f}%)")
    print(f"  overall: {correct.float().mean().item():.2%}")


def format_stats(stats):
    return "  |  ".join(
        f"digit {digit}: {s['pct_correct']:.2f}% correct / {s['pct_incorrect']:.2f}% incorrect"
        for digit, s in stats.items()
    )


def plot_pca(encoded, targets, true_labels, is_mislabeled, title, stats, ax):
    pca = PCA(n_components=2)
    stacked = torch.cat([encoded, targets], dim=0).numpy()
    pca.fit(stacked)

    encoded_2d = pca.transform(encoded.numpy())
    targets_2d = pca.transform(targets.numpy())

    true_labels = true_labels.numpy()
    is_mislabeled = is_mislabeled.numpy()
    # The digit each one-hot target vector actually encodes -- may differ from
    # true_labels for mislabeled samples, and is what determines star position.
    assigned_labels = targets.argmax(dim=1).numpy()

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

        # All targets assigned to this digit are the same one-hot vector, so a
        # single representative star is enough (and avoids draw-order overlap
        # between digits' stars hiding one another).
        target_mask = assigned_labels == digit
        if target_mask.any():
            target_point = targets_2d[target_mask][:1]
            ax.scatter(
                target_point[:, 0], target_point[:, 1],
                marker="*", s=250, color=color, edgecolors="black", linewidths=0.5,
                zorder=3, label=f"one-hot target, digit {digit}",
            )

    ax.set_title(f"{title}\n{format_stats(stats)}", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_pca(
    train_encoded, train["targets"], train["true_labels"], train["is_mislabeled"],
    "Train: encoder output vs. one-hot target (PCA)", train_stats, axes[0],
)
plot_pca(
    test_encoded, test["targets"], test["true_labels"], test["is_mislabeled"],
    "Test: encoder output vs. one-hot target (PCA)", test_stats, axes[1],
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig("pca_plots.png", dpi=150, bbox_inches="tight")
wandb.log({"pca_plots": wandb.Image(fig)})
plt.show()

run.finish()
