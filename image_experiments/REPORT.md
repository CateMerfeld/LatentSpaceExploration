# MNIST 6-vs-8 Mislabeled Encoder Experiment

## Objective

Train an encoder that maps MNIST images to 10-dimensional one-hot-style
encodings, using only two digits (6 and 8), where 20% of training labels are
deliberately corrupted (assigned to the *other* digit). The goal is to
visualize — via PCA — how label noise during training distorts the learned
encoding relative to the intended one-hot targets, and whether that distortion
shows up differently on clean, held-out data.

Files:
- `data_preprocessing.py` — loads and prepares the data
- `mnist_encoder_experiment.py` — model, training loop, W&B logging, PCA plots

## Data

- Source: `torchvision.datasets.MNIST`, filtered to digits 6 and 8 only
  (11,769 train images, 1,932 test images).
- Images are flattened to 784-dim vectors, scaled to `[0, 1]`.
- Targets are 10-dim one-hot vectors using **digit-matched indices**: digit 6
  → index 6, digit 8 → index 8. The other 8 dimensions are always 0. This
  keeps the unused dimensions interpretable and lets us see whether the model
  ever activates them (it shouldn't, since only two digits are in play).

### Mislabeling

For the **training split only**, 20% of each digit's samples have their
target flipped to the *other* digit's one-hot vector — the image itself is
untouched, only the label used for the loss is corrupted. The test split
always uses clean, true one-hot targets. `data_preprocessing.py` tracks, per
sample: `true_labels` (actual digit), `assigned_labels` (digit the target
encodes), `targets` (the one-hot vector itself), and `is_mislabeled`.

### Train / validation / test protocol

The mislabeled training set is further split 85/15 into `train`/`val`
(seeded, so reproducible). `val` is drawn from the same pool as `train`, but
its labels are then reset to the true, uncorrupted one-hot targets
(`clean_labels()` in `mnist_encoder_experiment.py`) — only the actual
`train` batches used in the loss carry the 20% corruption. This makes `val`
a clean, held-out-in-spirit signal during training (see below for why this
matters), while `train` remains the seeded 85/15 split originally reserved
for it. During training, only `train` and `val` are touched — `val_mse` is
logged every epoch purely for monitoring. **The test set is not referenced
anywhere inside the training loop.** Only after the loop finishes does the
script evaluate `test_mse` once, using the clean test labels, and log that
as a final summary value (not a per-epoch curve, since it's a single
post-hoc measurement).

## Model

Simple MLP encoder, `Encoder` in `mnist_encoder_experiment.py`:

```
Linear(784 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> 10) -> Sigmoid
```

The final `Sigmoid` bounds outputs to `[0, 1]`, matching the range of the
binary one-hot targets, which pairs naturally with MSE loss.

- Loss: MSE against the (possibly mislabeled) one-hot target
- Optimizer: Adam, lr=1e-3
- Batch size: 64, 500 epochs

## Experiment tracking (Weights & Biases)

Project: [`mnist-6-8-encoder`](https://wandb.ai/catemerfeld/mnist-6-8-encoder)

Each run logs:
- **Config**: digits used, mislabel fraction, val fraction, seed, epochs,
  batch size, learning rate, hidden dim, architecture string, loss/optimizer
  names, train/val/test sizes, parameter count.
- **Training curves**: `train_mse` and `val_mse` per epoch.
- **Weights/gradients**: histograms via `wandb.watch(model, log="all")`.
- **Final test performance**: `test_mse`, logged once after training as a
  summary value.
- **PCA comparison plot** (see below), logged as a media artifact, with
  per-digit classification accuracy embedded in each subplot's title.

Per-digit classification correct/incorrect counts are *not* logged to W&B —
they're computed and shown only in this report and in the PCA plot titles
(see "Classification accuracy" below).

Earlier versions of this experiment gave `val` the same corrupted labels as
`train` (a plain random subset, corruption included). That produced a
confusing pattern: `val_mse` climbed over training while `train_mse` fell,
even though `test_mse` (clean labels) ended up lowest of all — because as
the model learned to output the *true* digit identity, it drifted further
from `val`'s noisy targets even while learning the right thing, making `val`
useless as an early-stopping signal. Once `val` was switched to clean labels
(run [`4sl21eki`](https://wandb.ai/catemerfeld/mnist-6-8-encoder/runs/4sl21eki)),
`val_mse` tracks `test_mse` closely throughout training (`val_mse=0.0277`
vs. `test_mse=0.0281` at epoch 500) — confirming the corrupted-labels theory
and making `val_mse` a meaningful proxy for held-out performance.

## PCA visualization

For each split (train, test), the encoder's 10-dim outputs and the 10-dim
one-hot targets are jointly PCA-reduced to 2D (a PCA fit on the two stacked
side by side), then plotted together:

- Small filled dots: encoder output for correctly-labeled samples, colored by
  **true digit**.
- Hollow-outlined dots: encoder output for **mislabeled** samples (same true
  digit color) — these are the ones to watch, since they show whether the
  encoder still recovers the true digit despite training on a corrupted
  target.
- Star markers: the one-hot target location(s) for each digit.
- Subplot titles: per-digit classification accuracy (% correct / %
  incorrect, rounded to 2 decimal places) — see "Classification accuracy"
  below for how this is computed.

### Bug found and fixed: star markers on the train plot

Originally, the star markers were colored/grouped by `true_labels`, but their
*position* comes from `targets` — which for mislabeled train samples points at
the *other* digit's one-hot vector. That mismatch meant the "digit 6" (blue)
group plotted stars at both the 6- and 8-locations, and the "digit 8" (red)
group, drawn second, did the same — with red's overlapping draws painting over
blue at both locations (matplotlib draws later scatter calls on top). Net
effect: the train plot appeared to show two red stars and no blue one, even
though nothing was wrong with the model or training — it was purely a
plotting artifact. The test plot never showed this because test labels are
never corrupted, so `true_labels` and the target's encoded digit always agree
there.

Fix: star color/position is now derived from `targets.argmax(dim=1)` (i.e.
the digit the one-hot vector actually encodes), and only one representative
star is plotted per digit rather than one per sample (since all targets for a
given digit are identical vectors, and per-sample plotting is what caused the
draw-order overlap in the first place).

Confirmed after re-running: the train plot now shows exactly one blue star
(digit 6) and one red star (digit 8), matching the test plot's pattern.

## Classification accuracy (train + test)

The PCA plots show *where* encoder outputs land, but not directly how often
the model gets the digit right. To quantify that, each sample is given a
predicted digit by comparing the two relevant one-hot dimensions of the
encoder's output — whichever of dimension 6 or dimension 8 scores higher wins
(the other 8 output dimensions are never a training target, so they're
excluded rather than doing a full 10-way argmax). This prediction is derived
purely from the already-computed `train_encoded`/`test_encoded` outputs used
for the PCA plots — it doesn't change how those plot values are generated.
This breakdown is report-only (not logged to W&B), but the per-digit
percentages are also embedded directly in each PCA plot's title.

Results from the latest 500-epoch run, with clean `val` labels
([`4sl21eki`](https://wandb.ai/catemerfeld/mnist-6-8-encoder/runs/4sl21eki)),
`train_mse=0.0029`, `val_mse=0.0277`, `test_mse=0.0281`:

**Train** (includes the 20%-mislabeled samples; predictions are compared
against `true_labels`, not the corrupted targets):

| Digit | % Correct | % Incorrect |
|---|---|---|
| 6 | 81.51% | 18.49% |
| 8 | 81.70% | 18.30% |
| **Overall** | **81.61%** | 18.39% |

**Test** (clean labels, never seen during training):

| Digit | % Correct | % Incorrect |
|---|---|---|
| 6 | 85.28% | 14.72% |
| 8 | 83.57% | 16.43% |
| **Overall** | **84.42%** | 15.58% |

Both splits land in the low-to-mid 80s, with test outperforming train by
~3pp. Given 500 epochs drives `train_mse` down to 0.0029 while `test_mse`
sits at 0.0281 (a much smaller gap than before `val` was cleaned, and one
now corroborated by `val_mse` tracking `test_mse` closely — see above),
these figures reflect a model that has still meaningfully overfit to the
20%-corrupted training targets. With `val_mse` now a trustworthy signal, an
early-stopping run (e.g. around where `val_mse` bottoms out, roughly epoch
415-445 per the training log) would be a natural next experiment to check
whether it recovers accuracy closer to test's clean-label ceiling.

## Saved embeddings

After training, the encoder's outputs for the full train and test sets
(`train_encoded`, `test_encoded`) are saved to disk via `torch.save()`,
alongside `true_labels`, `assigned_labels`, and `is_mislabeled`, so the
classification/plotting logic above can be reproduced or extended without
re-running training.

## Known slowdown on this machine

Default PyTorch CPU threading caused a ~1000x slowdown here (1.3s/step vs.
1.3ms/step) due to thread contention, fixed with `torch.set_num_threads(1)`
at the top of the script.

## Status

- [x] Data pipeline with configurable mislabeling
- [x] MLP encoder trained with MSE
- [x] Train/val/test split with test isolated until after training
- [x] W&B logging of config, curves, weights, and final test metric
- [x] PCA visualization for train and test splits
- [x] Star-marker plotting bug fixed and confirmed via re-run
- [x] Per-digit train + test classification accuracy added (report + PCA
      plot titles, not logged to W&B)
- [x] Encoder embeddings saved to disk after training
- [x] `val` labels cleaned (no longer inherits train's 20% corruption),
      confirming val_mse now tracks test_mse
