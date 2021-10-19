import ast
import pickle
from os import path
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


def load_dataset(
        sampling_rate: int,
        in_path: str,
        out_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load raw waves and tabular data"""
    # Load and convert annotation data
    tabular = pd.read_csv(
        path.join(in_path, "ptbxl_database.csv"),
        index_col="ecg_id"
    )
    tabular.scp_codes = tabular.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(tabular, sampling_rate, in_path, out_path)
    return X, tabular


def load_raw_data_ptbxl(
        df: pd.DataFrame,
        sampling_rate: int,
        in_path: str,
        out_path: str) -> np.ndarray:
    """Collect all raw waves data together and caching it"""
    if sampling_rate not in [100, 500]:
        raise ValueError(
            f"Sampling rate must be 100 either 500, not {sampling_rate}!"
        )
    
    df_filename_column = (
        "filename_lr" if sampling_rate == 100
        else "filename_hr"
    )
    raw_data_path = path.join(out_path, "raw" + f"{sampling_rate}" + ".npy")

    if path.exists(raw_data_path):
        data = np.load(raw_data_path, allow_pickle=True)
    else:
        files = df[df_filename_column]
        data = [wfdb.rdsamp(path.join(in_path, file)) for file in tqdm(files)]
        data = np.array([np.transpose(signal) for signal, __ in data])

        with open(raw_data_path, "wb") as f:
            pickle.dump(data, f)

    return data


def compute_label_aggregations(
        tabular: pd.DataFrame,
        data_path: str) -> pd.DataFrame:
    """Aggregate diagnosis"""
    tabular["scp_codes_len"] = tabular.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(
        path.join(data_path, "scp_statements.csv"),
        index_col=0
    )

    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                diag_class = diag_agg_df.loc[key].diagnostic_class

                if str(diag_class) != "nan":
                    tmp.append(diag_class)

        return list(set(tmp))

    tabular["superdiagnostic"] = (
        tabular
        .scp_codes
        .apply(aggregate_diagnostic)
    )
    tabular["superdiagnostic_len"] = (
        tabular
        .superdiagnostic
        .apply(lambda x: len(x))
    )

    return tabular


def compute_labels(
        labels: np.ndarray,
        out_path: str) -> Tuple[np.ndarray, MultiLabelBinarizer]:
    """Convert multilabel to multi-hot"""
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    labels = mlb.transform(labels)

    # Cache MultiLabelBinarizer
    with open(path.join(out_path, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)
    
    return labels, mlb


def select_data(
    waves_: np.ndarray,
    tabular_: pd.DataFrame,
    out_path: str
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Select data with diagnosis"""
    waves = waves_[tabular_.superdiagnostic_len > 0]
    tabular = tabular_[tabular_.superdiagnostic_len > 0]
    labels, mlb = compute_labels(tabular.superdiagnostic.to_numpy(), out_path)

    return waves, tabular, labels, mlb.classes_


def load_processed_data(
        sampling_rate: int,
        raw_data_path: str,
        interim_data_path: str,
        proc_data_path: str
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load ptbxl data"""

    print("Loading PTBXL...")
    print(f"Loading raw data with sampling rate {sampling_rate}...")
    waves, raw_labels = load_dataset(
        sampling_rate, raw_data_path, interim_data_path
    )

    print("Computing label aggregations...")
    # Preprocess label data
    tabular = compute_label_aggregations(raw_labels, raw_data_path)

    print("Selecting data...")
    # Select relevant data and convert to one-hot
    waves, tabular, labels, classes = select_data(
        waves, tabular, out_path=proc_data_path
    )

    return waves, tabular, labels, classes
