from os import path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ds.load_data import load_processed_data


class PtbXlWrapper:
    """Store train/test/validation waves and tabulardata"""

    def __init__(
        self, data_dir: str, sampling_rate: int = 100, batch_size: int = 64
    ) -> None:
        """Separate data into train/test/val datasets"""
        raw_data_path = path.join(data_dir, "raw")
        interim_data_path = path.join(data_dir, "interim")
        proc_data_path = path.join(data_dir, "processed")

        self.sampling_rate = sampling_rate
        self.batch_size = batch_size

        waves, tabular, labels, classes = load_processed_data(
            sampling_rate, raw_data_path, interim_data_path, proc_data_path
        )

        self.tabular = tabular
        self.classes = classes

        # Standardize our features
        X_waves_train, X_waves_test, X_waves_val = standardize_xs(
            waves[tabular.strat_fold < 9],
            waves[tabular.strat_fold == 9],
            waves[tabular.strat_fold == 10],
        )

        # 1-8 for training
        self.X_waves_train = X_waves_train
        self.y_train = labels[tabular.strat_fold < 9]

        # 9 for test
        self.X_waves_test = X_waves_test
        self.y_test = labels[tabular.strat_fold == 9]

        # 10 for validation
        self.X_waves_val = X_waves_val
        self.y_val = labels[tabular.strat_fold == 10]

    def make_train_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.X_waves_train, self.y_train), batch_size=self.batch_size
        )

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.X_waves_test, self.y_test), batch_size=self.batch_size
        )

    def make_val_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self.X_waves_val, self.y_val), batch_size=self.batch_size
        )


class PtbXl(Dataset):
    """Implement ptbxl dataset"""

    features: torch.Tensor
    labels: torch.Tensor
    index: int
    length: int

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
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
            torch.tensor(label, dtype=torch.float32),
        )


def standardize_xs(
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray]:
    """
    Apply standart scalar
    Fit it to train data and transfrom all dataset
    """
    ss = StandardScaler()
    ss.fit(np.vstack(x_train).flatten()[:, np.newaxis].astype(float))

    return (
        apply_standardizer(x_train, ss),
        apply_standardizer(x_test, ss),
        apply_standardizer(x_val, ss),
    )


def apply_standardizer(features: np.ndarray, ss: StandardScaler) -> np.ndarray:
    features_tmp = []
    for x in features:
        x_shape = x.shape
        features_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))

    return np.array(features_tmp)
