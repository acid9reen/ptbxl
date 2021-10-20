from os import path

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ds.load_data import load_processed_data


class PtbXlWrapper:
    """Store train/test/validation waves and tabulardata """
    def __init__(
            self,
            data_dir: str,
            sampling_rate: int=100,
            batch_size: int = 64) -> None:
        """Separate data into train/test/val datasets"""
        raw_data_path = path.join(data_dir, "raw")
        interim_data_path = path.join(data_dir, "interim")
        proc_data_path = path.join(data_dir, "processed")

        self.sampling_rate = sampling_rate
        self.batch_size = batch_size

        waves, tabular, labels, classes = load_processed_data(
            sampling_rate,
            raw_data_path,
            interim_data_path,
            proc_data_path
        )

        self.tabular = tabular
        self.classes = classes

        # 1-8 for training
        self.X_waves_train = waves[tabular.strat_fold < 9]
        self.y_train = labels[tabular.strat_fold < 9]

        # 9 for test
        self.X_waves_test = waves[tabular.strat_fold == 9]
        self.y_test = labels[tabular.strat_fold == 9]

        # 10 for validation
        self.X_waves_val = torch.tensor(
            waves[tabular.strat_fold == 10],
            dtype=torch.float32
        )
        self.y_val = torch.tensor(
            labels[tabular.strat_fold == 10],
            dtype=torch.float32
        )

    def make_train_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.X_waves_train, self.y_train),
            batch_size=self.batch_size
        )

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.X_waves_test, self.y_test),
            batch_size=self.batch_size
        )


class PtbXl(Dataset):
    """Implement ptbxl dataset"""
    features: torch.Tensor
    labels: torch.Tensor
    index: int
    length: int

    def __init__(
            self,
            features: torch.Tensor,
            labels: torch.Tensor) -> None:
        """Create dataset from user features and labels"""
        feat_len, labels_len = len(features), len(labels)
        if feat_len != labels_len:
            raise ValueError(
                f"Length of features and labels must be the same,"
                f"but it's {feat_len}, {labels_len} respectively"
            )
        self.length = feat_len
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )
