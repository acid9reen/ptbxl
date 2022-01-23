import torch

from ds.dataset import PtbXlWrapper
from ds.tensorboard import TensorboardExperiment
from ds.runner import Runner, run_epoch, run_validation
from ds.models import ConvLinearBasicModel


# Hyperparameters
EPOCH_COUNT = 20
LR = 8e-4
BATCH_SIZE = 128
LOG_PATH = "./runs"

# Data confuguration
DATA_DIR = "data"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}")


def main():
    sampling_rate = 100

    dataset = PtbXlWrapper(DATA_DIR, sampling_rate, BATCH_SIZE)

    # Create the data loaders
    train_dl = dataset.make_train_dataloader()
    test_dl = dataset.make_test_dataloader()
    val_dl = dataset.make_val_dataloader()

    # Model and optimizer
    model = ConvLinearBasicModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the runners
    test_runner = Runner(test_dl, model, device=DEVICE)
    train_runner = Runner(train_dl, model, optimizer, device=DEVICE)
    val_runner = Runner(val_dl, model, device=DEVICE)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(LOG_PATH)

    # Run the epochs
    for epoch_id in range(EPOCH_COUNT):
        run_epoch(test_runner, train_runner, tracker, epoch_id, dataset.classes)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
                f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()

    run_validation(val_runner, tracker, dataset.classes)


if __name__ == "__main__":
    main()
