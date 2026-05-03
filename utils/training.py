"""
Training and Evaluation Utilities for CIFAR-10.

Provides reusable functions for training loops, validation, testing, and
per-class accuracy computation — all with AMP (mixed precision) support.
"""

import time
import torch


def train_one_epoch(
        model, loader, criterion, optimizer,
        scheduler, scaler, device
):
    """
    Run one full training epoch with AMP.

    Args:
        model: The nn.Module to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        scheduler: LR scheduler (stepped per mini-batch, e.g. OneCycleLR).
        scaler: torch.amp.GradScaler for mixed precision.
        device: Target device ('cuda' or 'cpu').

    Returns:
        avg_train_loss (float): Mean training loss over all batches.
    """
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR steps per mini-batch
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / n_batches


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataset (validation or test).

    Args:
        model: The nn.Module to evaluate.
        loader: DataLoader for evaluation.
        criterion: Loss function.
        device: Target device.

    Returns:
        (avg_loss, accuracy_pct): Tuple of average loss and accuracy as a percentage.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def test_accuracy(model, loader, classes, device):
    """
    Compute overall and per-class accuracy on a test set.

    Args:
        model: The nn.Module to evaluate.
        loader: Test DataLoader.
        classes: Tuple of class name strings.
        device: Target device.

    Returns:
        (overall_acc, per_class_dict): Overall accuracy (%) and a dict
        mapping class names to their individual accuracy (%).
    """
    model.eval()
    correct = 0
    total = 0
    correct_pred = {c: 0 for c in classes}
    total_pred = {c: 0 for c in classes}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    overall_acc = 100.0 * correct / total
    per_class = {
        name: 100.0 * correct_pred[name] / total_pred[name]
        for name in classes
    }

    return overall_acc, per_class


def fit(
    model, trainloader, valloader, criterion,
        optimizer, scheduler, scaler,
    device, epochs, checkpoint=None,
):
    """
    Full training loop with per-epoch logging, validation, and optional checkpointing.

    Args:
        model: The nn.Module to train.
        trainloader: Training DataLoader.
        valloader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler (per mini-batch).
        scaler: AMP GradScaler.
        device: Target device.
        epochs: Number of epochs to train.
        checkpoint: Optional ModelCheckpoint callback.

    Returns:
        (train_losses, val_losses, val_accuracies): Lists of per-epoch metrics.
    """
    train_losses, val_losses, val_accuracies = [], [], []
    total_start = time.time()

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        avg_train_loss = train_one_epoch(
            model, trainloader, criterion, optimizer, scheduler, scaler, device,
        )
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss, val_acc = evaluate(model, valloader, criterion, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Checkpoint
        if checkpoint is not None:
            checkpoint(val_acc, epoch, optimizer, scheduler, scaler)

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time
        print(
            f'Epoch [{epoch + 1:2d}/{epochs}]  '
            f'Train Loss: {avg_train_loss:.4f}  '
            f'Val Loss: {avg_val_loss:.4f}  '
            f'Val Acc: {val_acc:.2f}%  '
            f'LR: {current_lr:.6f}  '
            f'Time: {elapsed:.2f}s'
        )

    total_minutes = (time.time() - total_start) / 60
    print(f'\nFinished Training in {total_minutes:.2f} minutes')
    if checkpoint is not None:
        print(f'Best Validation Accuracy: {checkpoint.best_score:.2f}%')

    return train_losses, val_losses, val_accuracies
