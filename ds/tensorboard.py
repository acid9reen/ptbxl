from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter

from ds.tracking import Stage
from ds.utils import create_experiment_log_dir


class TensorboardExperiment:
    def __init__(self, log_path: str, create: bool = True):

        log_dir = create_experiment_log_dir(root=log_path)
        self.stage = Stage.TRAIN
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_confusion_matrix(
        self,
        y_true: list[np.array],
        y_pred: list[np.array],
        step: int,
        classes: tuple[str],
    ) -> None:
        y_true, y_pred = self.collapse_batches(y_true, y_pred)
        fig = self.create_confusion_matrix(y_true, y_pred, step, classes)
        tag = f"{self.stage.name}/epoch/confusion_matrix"
        self._writer.add_figure(tag, fig, step)

    @staticmethod
    def collapse_batches(
        y_true: list[np.array], y_pred: list[np.array]
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(y_true), np.concatenate(y_pred)

    def create_confusion_matrix(
        self,
        y_true: list[np.array],
        y_pred: list[np.array],
        step: int,
        classes: tuple[str],
    ) -> plt.Figure:
        cfs = multilabel_confusion_matrix(y_true, y_pred)
        fig, axes = plt.subplots(1, len(cfs), figsize=(20, 7), dpi=87)
        axes = axes.ravel()

        for axe, cf, title in zip(axes, cfs, classes):
            disp = ConfusionMatrixDisplay(cf)
            disp.plot(ax=axe)
            disp.im_.colorbar.remove()
            disp.ax_.set_title(title)

        return fig
