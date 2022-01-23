from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ds.metrics import Accuracy
from ds.tracking import ExperimentTracker, Stage


def activate(x: float, threshold: float = 0.5) -> float:
    return 1 if x > threshold else 0


v_activate = np.vectorize(activate)


class Runner:
    def __init__(
        self,
        loader: DataLoader,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> None:
        self.run_count = 0
        self.loader = loader
        self.accuracy_metric = Accuracy()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        # Objective (loss) function
        self.compute_loss = torch.nn.BCELoss(reduction="sum")
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        # Assume Stage based on presence of optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):
        self.model.train(self.stage is Stage.TRAIN)

        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            x, y = x.to(self.device), y.to(self.device)
            loss, batch_accuracy = self._run_single(x, y)

            experiment.add_batch_metric("Accuracy", batch_accuracy, self.run_count)

            if self.optimizer:
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Test Metrics
        y_np = y.cpu().detach().numpy()
        y_prediction_np = prediction.cpu().detach().numpy()
        y_prediction_np_activ = np.zeros(y_prediction_np.shape)

        for ind, arr in enumerate(y_prediction_np, 0):
            y_prediction_np_activ[ind] = v_activate(arr, 0.5)

        batch_accuracy: float = accuracy_score(y_np, y_prediction_np_activ)
        self.accuracy_metric.update(batch_accuracy, batch_size)

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np_activ]
        return loss, batch_accuracy

    def reset(self):
        self.accuracy_metric = Accuracy()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(
    test_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    epoch_id: int,
    classes: tuple[str],
):
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    # Log Training Epoch Metrics
    experiment.add_epoch_metric("Accuracy", train_runner.avg_accuracy, epoch_id)

    # Testing Loop
    experiment.set_stage(Stage.TEST)
    test_runner.run("Test Batches", experiment)

    # Log Test Epoch Metrics
    experiment.add_epoch_metric("Accuracy", test_runner.avg_accuracy, epoch_id)
    precision, recall, f1_score, __ = precision_recall_fscore_support(
        np.concatenate(test_runner.y_true_batches),
        np.concatenate(test_runner.y_pred_batches),
        average="samples",
        zero_division=0,
    )
    experiment.add_epoch_metric("Precision", precision, epoch_id)
    experiment.add_epoch_metric("Recall", recall, epoch_id)
    experiment.add_epoch_metric("f1_score", f1_score, epoch_id)
    experiment.add_epoch_confusion_matrix(
        test_runner.y_true_batches, test_runner.y_pred_batches, epoch_id, classes
    )


def run_validation(
    val_runner: Runner, experiment: ExperimentTracker, classes: tuple[str]
):
    epoch_id = 0

    experiment.set_stage(Stage.VAL)
    val_runner.run("Val Batches", experiment)

    experiment.add_epoch_metric("Accuracy", val_runner.avg_accuracy, epoch_id)
    precision, recall, f1_score, __ = precision_recall_fscore_support(
        np.concatenate(val_runner.y_true_batches),
        np.concatenate(val_runner.y_pred_batches),
        average="samples",
        zero_division=0,
    )
    experiment.add_epoch_metric("Precision", precision, epoch_id)
    experiment.add_epoch_metric("Recall", recall, epoch_id)
    experiment.add_epoch_metric("f1_score", f1_score, epoch_id)
    experiment.add_epoch_confusion_matrix(
        val_runner.y_true_batches, val_runner.y_pred_batches, epoch_id, classes
    )
