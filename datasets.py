from create_data import create_unconditional_olympic_rings, create_olympic_rings
from torch.utils.data import Dataset, DataLoader
import torch


def get_dataloaders(cfg):
    if cfg.data.conditioned == "normalizing_flow":
        train = DataLoader(UnconditionalOlympicRings(cfg.data.number_of_data_points), batch_size=cfg.trainer.batch_size,
                           shuffle=True)
        test = DataLoader(UnconditionalOlympicRings(cfg.data.number_of_data_points // 10),
                          batch_size=cfg.trainer.batch_size, shuffle=True)
        return train, test
    else:
        train = DataLoader(ConditionalOlympicRings(cfg.data.number_of_data_points), batch_size=cfg.trainer.batch_size,
                           shuffle=True)
        test = DataLoader(ConditionalOlympicRings(cfg.data.number_of_data_points // 10),
                          batch_size=cfg.trainer.batch_size, shuffle=True)
        return train, test


class UnconditionalOlympicRings(Dataset):
    def __init__(self, n_points):
        data = create_unconditional_olympic_rings(n_points=n_points)
        self.data = torch.tensor(data, dtype=torch.float64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConditionalOlympicRings(Dataset):
    def __init__(self, n_points):
        data, labels, legend = create_olympic_rings(n_points=n_points)
        self.data = torch.tensor(data, dtype=torch.float64)
        self.labels = torch.tensor(labels, dtype=torch.int)
        self.legend = legend

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
