from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Logging function that records training loss, training accuracy, and validation accuracy.
    """

    global_step = 0  # Ensure correct global step tracking

    for epoch in range(10):
        metrics = {"train_accuracy": [], "val_accuracy": []}

        # Example training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # Log training loss at every iteration
            logger.add_scalar('train_loss', dummy_train_loss, global_step=global_step)

            # Collect training accuracy values
            avg_train_acc = dummy_train_accuracy.mean().item()
            metrics["train_accuracy"].append(avg_train_acc)

            global_step += 1  # Increment global step

        # Log average train_accuracy **with global_step at the last training step of the epoch**
        avg_train_accuracy = sum(metrics["train_accuracy"]) / len(metrics["train_accuracy"])
        logger.add_scalar('train_accuracy', avg_train_accuracy, global_step=global_step)

        # Example validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)
            metrics["val_accuracy"].append(dummy_validation_accuracy.mean().item())

        # Log average validation accuracy **at the end of the epoch**
        avg_val_accuracy = sum(metrics["val_accuracy"]) / len(metrics["val_accuracy"])
        logger.add_scalar('val_accuracy', avg_val_accuracy, global_step=global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
