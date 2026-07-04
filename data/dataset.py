import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data


class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def create_dataloaders(data, targets, batch_size=32, shuffle=True, pin_memory=True, num_workers=0):
    dataset = SimpleDataset(data, targets)
    use_pin = pin_memory and torch.cuda.is_available() and num_workers > 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_pin,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    return dataloader
