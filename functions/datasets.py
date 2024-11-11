from torch.utils.data import Dataset
import torch

from typing import List


class PolarsDataset(Dataset):
    def __init__(
        self, dataframe, label_column: str, feature_columns: List[str]
    ):
        self.data = dataframe.to_numpy()

        self.features = torch.FloatTensor(
            dataframe.select(feature_columns).to_numpy()
        )
        self.labels = torch.FloatTensor(
            dataframe.select(label_column).to_numpy().flatten()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PolarsCollabDataset(Dataset):
    def __init__(
        self, dataframe, label_column: str, user_column: str, item_column: str
    ):
        self.data = dataframe.to_numpy()

        self.user_ids = torch.IntTensor(
            dataframe.select(user_column).to_numpy().flatten()
        )

        self.item_ids = torch.IntTensor(
            dataframe.select(item_column).to_numpy().flatten()
        )

        self.labels = torch.FloatTensor(
            dataframe.select(label_column).to_numpy().flatten()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
