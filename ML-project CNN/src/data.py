"""
CIFAR-10 data loading with augmentation for training.
"""

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class TransformSubset(Dataset):
    """
    Wraps a Subset and applies its own transform, independent of the
    underlying dataset's transform.  This avoids the bug where writing
    val_set.dataset.transform = ... also mutates the training set's
    transform (both share the same CIFAR10 dataset object).
    """

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset.dataset.data[self.subset.indices[idx]], \
                     self.subset.dataset.targets[self.subset.indices[idx]]
        # dataset.data is a NumPy array (H, W, C); convert to PIL for transforms
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Splits the official 50k training set into train/val.

    The val split uses test-time transforms (no augmentation).
    The train split uses training transforms (crop, flip, jitter).
    Both operate on independent transform pipelines — mutating one
    does NOT affect the other.
    """
    # Load raw training data without any transform; transforms are applied
    # per-split via TransformSubset so they never interfere with each other.
    train_data_raw = datasets.CIFAR10(data_dir, train=True, download=True, transform=None)
    test_data      = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=get_transforms(False))

    # Deterministic train/val split
    n_val   = int(len(train_data_raw) * val_split)
    n_train = len(train_data_raw) - n_val
    train_subset, val_subset = random_split(
        train_data_raw, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Wrap each split with its own transform — no shared state
    train_set = TransformSubset(train_subset, get_transforms(train=True))
    val_set   = TransformSubset(val_subset,   get_transforms(train=False))

    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_set, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
