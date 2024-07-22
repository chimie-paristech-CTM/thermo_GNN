from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch

class MoleculeDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.smiles = self.data['smiles']
        self.features = self.data.drop(columns=['smiles', 'energy'])
        self.targets = self.data['energy']

        # 检查数据中的 NaN 和无穷大值
        assert not self.features.isnull().values.any(), "Features contain NaN"
        assert not self.targets.isnull().values.any(), "Targets contain NaN"
        assert np.isfinite(self.features.values).all(), "Features contain infinite values"
        assert np.isfinite(self.targets.values).all(), "Targets contain infinite values"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)
        smiles = self.smiles.iloc[idx]
        return features, target, smiles