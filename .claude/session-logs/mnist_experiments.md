# mnist_experiments

## Goal
Build an encoder mapping MNIST digits 6 and 8 to 10-dim one-hot-style
encodings via MSE loss, with 20% of training labels deliberately mislabeled
(flipped to the other digit). Visualize encoder output vs. one-hot targets
via PCA for both train and test splits, track everything in W&B, and
document the experiment in a report.

## Key files
- `image_experiments/data_preprocessing.py` — loads MNIST, filters to
  digits 6/8, builds one-hot targets, applies 20% train-only mislabeling.
  Unchanged since creation.
- `image_experiments/mnist_encoder_experiment.py` — main script: MLP encoder
  (784→128→64→10, Sigmoid), train/val/test split (val carved from train,
  then given clean labels via `clean_labels()`; test untouched until after
  training), W&B logging (no longer logs a redundant `epoch` metric),
  embeddings saved to scratch, per-digit classification stats, PCA plots
  with per-digit accuracy in titles.
- `image_experiments/REPORT.md` — written report of the experiment. Updated
  with run `4sl21eki`'s numbers (train + test classification, val-cleaning
  change, saved embeddings, W&B scope).
- `.gitignore` — added `__pycache__/`, `*.pyc`, `data/`, `*.png`, `wandb/`
  after an unauthorized/unexplained commit (`6f1dce1`) was found on `main`
  containing 64MB of MNIST binaries; cleaned up in commit `33f4c5f`
  (git rm --cached + .gitignore). That cleanup commit was never pushed —
  this sandbox has no git push credentials.

## Key decisions / tradeoffs
- One-hot targets use digit-matched indices (digit 6 → index 6, digit 8 →
  index 8), not compact 0/1 indices — keeps unused dims interpretable.
- `torch.set_num_threads(1)` is required near the top of the script — this
  sandbox has a ~1000x CPU thread-contention slowdown without it
  (1339ms/step → 1.3ms/step).
- Star markers in PCA plots must be colored/positioned by
  `targets.argmax(dim=1)` (assigned label), not `true_labels` — using
  true_labels caused a draw-order bug where mislabeled train stars painted
  over each other (fixed, plus `zorder=3` so stars always render above dots).
- Per-digit classification (correct/incorrect %) is computed read-only from
  the already-existing `train_encoded`/`test_encoded` tensors — never
  changes how those tensors or the PCA plot are generated. User was explicit
  about this boundary.
- Correct/incorrect breakdown is report-only, not logged to W&B (per user
  request) — W&B keeps config, train/val MSE curves, weight histograms,
  final test_mse, and the PCA plot image.
- Predicted embeddings are saved via `torch.save()` to the harness scratch
  dir (`/tmp/claude-1003/.../scratchpad/mnist_6_8_embeddings.pt`), not
  committed to the repo.
- `val` is now given clean (uncorrupted) labels after being split from
  `train`, via a new `clean_labels()` helper — only `train`'s actual loss
  batches carry the 20% mislabeling now. This was a direct fix for `val_mse`
  being useless as a monitoring signal (it used to climb throughout training
  even as the model improved, since it inherited train's corrupted targets).
  After the fix, `val_mse` tracks `test_mse` closely, confirming the
  root-cause theory.
- Removed the redundant `epoch` key from `wandb.log()` — W&B's own step
  counter already served that purpose, so logging `epoch` just produced a
  useless diagonal-line chart (epoch vs. step, both incrementing by 1/call).

## Current state
Latest successful run: `4sl21eki`
(https://wandb.ai/catemerfeld/mnist-6-8-encoder/runs/4sl21eki), val labels
now clean.
- train_mse=0.0029, val_mse=0.0277, test_mse=0.0281 (val now tracks test
  closely, unlike the old corrupted-val runs)
- train classification: digit 6 81.51% correct / 18.49% incorrect,
  digit 8 81.70% / 18.30%, overall 81.61%
- test classification: digit 6 85.28% / 14.72%, digit 8 83.57% / 16.43%,
  overall 84.42%
- Embeddings saved to scratch; PCA plot titles show 2-decimal per-digit
  pct correct/incorrect.

`REPORT.md` now reflects `4sl21eki` throughout: updated train/val/test
protocol section (val-cleaning explained), rewritten val_mse-behavior
narrative (previously attributed to corrupted val labels — now confirmed
since val_mse tracks test_mse post-fix), updated classification tables,
and an updated status checklist.

## Open questions / next steps
- Cleanup commit `33f4c5f` (the `.gitignore`/git rm fix) is still unpushed;
  user needs to push it from an environment with git credentials if desired.
- Possible next experiment (noted in REPORT.md but not yet run):
  early-stopping around epoch 415-445, where val_mse bottoms out in the
  4sl21eki run, to see if it recovers accuracy closer to test's ceiling.
