#%%
import torch
from torchvision import datasets, transforms

DATA_ROOT = "./data"


def load_mnist_digits(digits=(6, 8), mislabel_frac=0.2, seed=42, data_root=DATA_ROOT):
    """
    Loads MNIST, keeps only the given digits, and builds 10-dim one-hot
    targets (digit-matched indices, so most dims are always 0).

    In the train split, `mislabel_frac` of each digit's samples get their
    target flipped to the *other* digit's one-hot vector (the image itself
    is untouched). The test split always uses clean/true one-hot targets.

    Returns a dict with, for each of "train"/"test":
      images: (N, 784) float tensor in [0, 1]
      true_labels: (N,) long tensor, true digit value
      assigned_labels: (N,) long tensor, digit value the target one-hot encodes
      targets: (N, 10) float tensor, one-hot (possibly mislabeled for train)
      is_mislabeled: (N,) bool tensor
    """
    generator = torch.Generator().manual_seed(seed)
    to_tensor = transforms.ToTensor()

    raw_train = datasets.MNIST(data_root, train=True, download=True, transform=to_tensor)
    raw_test = datasets.MNIST(data_root, train=False, download=True, transform=to_tensor)

    digits = list(digits)
    other_digit = {digits[0]: digits[1], digits[1]: digits[0]}

    def filter_and_flatten(raw_dataset):
        images = raw_dataset.data.float() / 255.0
        labels = raw_dataset.targets
        mask = torch.isin(labels, torch.tensor(digits))
        images = images[mask].reshape(mask.sum(), -1)
        labels = labels[mask]
        return images, labels

    def make_split(images, true_labels, mislabel):
        n = true_labels.shape[0]
        is_mislabeled = torch.zeros(n, dtype=torch.bool)
        assigned_labels = true_labels.clone()

        if mislabel:
            for digit in digits:
                digit_idx = (true_labels == digit).nonzero(as_tuple=True)[0]
                n_flip = int(round(mislabel_frac * digit_idx.shape[0]))
                perm = digit_idx[torch.randperm(digit_idx.shape[0], generator=generator)]
                flip_idx = perm[:n_flip]
                assigned_labels[flip_idx] = other_digit[digit]
                is_mislabeled[flip_idx] = True

        targets = torch.zeros(n, 10)
        targets[torch.arange(n), assigned_labels] = 1.0

        return {
            "images": images,
            "true_labels": true_labels,
            "assigned_labels": assigned_labels,
            "targets": targets,
            "is_mislabeled": is_mislabeled,
        }

    train_images, train_labels = filter_and_flatten(raw_train)
    test_images, test_labels = filter_and_flatten(raw_test)

    return {
        "train": make_split(train_images, train_labels, mislabel=True),
        "test": make_split(test_images, test_labels, mislabel=False),
    }


#%%
if __name__ == "__main__":
    data = load_mnist_digits()
    for split_name, split in data.items():
        n = split["true_labels"].shape[0]
        n_mislabeled = split["is_mislabeled"].sum().item()
        print(f"{split_name}: {n} samples, {n_mislabeled} mislabeled ({n_mislabeled / n:.1%})")
