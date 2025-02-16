from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Your code here - finish logging the dummy loss and accuracy

    For training, log the training loss every iteration and the average accuracy every epoch
    Call the loss 'train_loss' and accuracy 'train_accuracy'

    For validation, log only the average accuracy every epoch
    Call the accuracy 'val_accuracy'

    Make sure the logging is in the correct spot so the global_step is set correctly,
    for epoch=0, iteration=0: global_step=0
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # example training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # Log train_loss
            logger.add_scalar('train_loss', dummy_train_loss, global_step)
            
            # Save additional metrics to be averaged
            metrics["train_acc"].append(dummy_train_accuracy.mean().item())

            global_step += 1

        # Log average train_accuracy
        avg_train_accuracy = sum(metrics["train_acc"]) / len(metrics["train_acc"])
        logger.add_scalar('train_accuracy', avg_train_accuracy, epoch)

        # example validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)

             # Save additional metrics to be averaged
            metrics["val_acc"].append(dummy_validation_accuracy.mean().item())

        # Log average val_accuracy
        avg_val_accuracy = sum(metrics["val_acc"]) / len(metrics["val_acc"])
        logger.add_scalar('val_accuracy', avg_val_accuracy, epoch)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
