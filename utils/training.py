"""
Training and Evaluation Utilities for CIFAR-10 / CIFAR-100.

Provides reusable functions for training loops, validation, testing, and
per-class accuracy computation — all with AMP (mixed precision) support.

Optional Features:
    MixUp augmentation: pass ``mixup_alpha > 0`` to ``train_one_epoch`` / ``fit``.
    RunLogger:          set ``log=True`` in ``fit()`` for zero-config experiment tracking.
"""

import inspect
import time
from pathlib import Path

import numpy as np
import torch


def _caller_log_dir() -> str:
    """
    Walk the call stack to find the calling script or notebook name,
    then return an absolute log directory path derived from that file's stem.
    Logs are stored in a centralized directory at the repository root.

    Handles all common execution environments:
        - Regular .py scripts          → {project_root}/logs/{script_stem}/
        - VS Code Jupyter notebooks    → {project_root}/logs/{notebook_stem}/  (__vsc_ipynb_file__)
        - PyCharm / classic Jupyter    → {project_root}/logs/{notebook_stem}/  (__file__ or co_filename)
        - Unknown / inline cells       → {project_root}/logs/runs/
    """
    utils_dir = Path(__file__).parent.resolve()
    project_root = utils_dir.parent
    frame = inspect.currentframe()
    try:
        while frame is not None:
            fname  = frame.f_code.co_filename
            globs  = frame.f_globals

            # ── Notebook detection (globals-based) ────────────────────────────
            # VS Code Jupyter injects __vsc_ipynb_file__ into every cell's globals
            vsc_nb = globs.get("__vsc_ipynb_file__")
            if vsc_nb:
                return str(project_root / "logs" / Path(vsc_nb).stem)

            # PyCharm / classic Jupyter may set __file__ to the .ipynb path
            file_glob = globs.get("__file__", "")
            if file_glob.endswith(".ipynb"):
                return str(project_root / "logs" / Path(file_glob).stem)

            # ── Filename-based detection ──────────────────────────────────────
            # Newer kernels (JupyterLab ≥ 4, nbclient) set co_filename to the
            # actual notebook path instead of <ipython-input-…>
            if fname.endswith(".ipynb"):
                return str(project_root / "logs" / Path(fname).stem)

            # Regular .py file outside this utils package
            if (not fname.startswith("<")
                    and utils_dir not in Path(fname).resolve().parents):
                return str(project_root / "logs" / Path(fname).stem)

            frame = frame.f_back
    finally:
        del frame   # avoid reference cycle

    return str(project_root / "logs" / "runs")


# ── MixUp Utilities ──────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Apply MixUp augmentation to a mini-batch.

    Samples a mixing ratio ``lam`` from Beta(alpha, alpha), then linearly
    interpolates inputs between two random shuffled samples in the batch.

    Args:
        x:      Input tensor  (B, C, H, W).
        y:      Label tensor  (B,)  — hard integer labels.
        alpha:  Beta distribution concentration parameter.
                Typical values: 0.2 (mild), 0.4 (moderate), 1.0 (strong).
        device: Target device string used for the permutation index.

    Returns:
        mixed_x (Tensor), y_a (Tensor), y_b (Tensor), lam (float)
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the convex combination of two cross-entropy losses for MixUp.

    Args:
        criterion: Loss function (e.g. nn.CrossEntropyLoss).
        pred:      Model logits (B, num_classes).
        y_a:       First label set  (B,).
        y_b:       Second label set (B,).
        lam:       Mixing ratio from ``mixup_data``.

    Returns:
        Scalar loss tensor.
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ── Core Training Functions ───────────────────────────────────────────────────


def train_one_epoch(
        model, loader, criterion, optimizer,
        scheduler, scaler, device, mixup_alpha=0.0,
        step_scheduler_per_batch=True,
):
    """
    Run one full training epoch with AMP and optional MixUp augmentation.

    Args:
        model:        The nn.Module to train.
        loader:       Training DataLoader.
        criterion:    Loss function (e.g. nn.CrossEntropyLoss).
        optimizer:    Optimizer instance.
        scheduler:    LR scheduler.
        scaler:       torch.amp.GradScaler for mixed precision.
        device:       Target device ('cuda' or 'cpu').
        mixup_alpha:  Beta distribution parameter for MixUp. 0 disables MixUp;
                      recommended range 0.2–0.4 for CIFAR-100.
        step_scheduler_per_batch: If True, steps scheduler every mini-batch (e.g., OneCycleLR).
                                  If False, scheduler is stepped externally per epoch.

    Returns:
        avg_train_loss (float): Mean training loss over all batches.
    """
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    use_mixup = mixup_alpha > 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, labels, alpha=mixup_alpha, device=device
            )

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per mini-batch if requested
        if scheduler is not None and step_scheduler_per_batch:
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
        mixup_alpha=0.0, step_scheduler_per_batch=True,
        log=False, log_dir=None,
):
    """
    Full training loop with per-epoch logging, validation, and optional
    MixUp augmentation.

    Args:
        model:          The nn.Module to train.
        trainloader:    Training DataLoader.
        valloader:      Validation DataLoader.
        criterion:      Loss function.
        optimizer:      Optimizer.
        scheduler:      LR scheduler.
        scaler:         AMP GradScaler.
        device:         Target device string.
        epochs:         Number of epochs to train.
        checkpoint:     Optional ModelCheckpoint callback.
        mixup_alpha:    Beta parameter for MixUp (0 = disabled).
                        Recommended: 0.2–0.4 for CIFAR-100.
        step_scheduler_per_batch: If True (default), scheduler steps per mini-batch (OneCycleLR).
                                  If False, steps per epoch (CosineAnnealingLR).
        log:            If True, automatically create a RunLogger and record this run.
                        Each run is saved as its own JSON file — nothing is ever overwritten.
        log_dir:        Directory in which run files are saved.  When ``None`` (default),
                        derived from the calling script/notebook name::

                            ImageClassifierTorch_ResNet.ipynb
                              → logs/ImageClassifierTorch_ResNet/
                            train.py
                              → logs/train/

    Returns:
        (train_losses, val_losses, val_accuracies): Lists of per-epoch metrics.
    """
    # ── Set up logger ─────────────────────────────────────────────────────────
    logger = None
    if log:
        from .logger import RunLogger
        resolved_log_dir = log_dir if log_dir is not None else _caller_log_dir()
        # verbose=False: fit() already prints per-epoch lines; logger records silently
        logger = RunLogger(log_dir=resolved_log_dir, verbose=False)
        config = {
            "model":       type(model).__name__,
            "optimizer":   type(optimizer).__name__,
            "scheduler":   type(scheduler).__name__ if scheduler else None,
            "lr":          optimizer.param_groups[0]["lr"],
            "epochs":      epochs,
            "mixup_alpha": mixup_alpha,
            "device":      str(device),
        }
        logger.start(config=config)
        print(f"[RunLogger] Logging to '{resolved_log_dir}/'  —  run: '{logger.run_name}'")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses, val_accuracies = [], [], []
    total_start = time.time()

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        avg_train_loss = train_one_epoch(
            model, trainloader, criterion, optimizer, scheduler, scaler,
            device, mixup_alpha=mixup_alpha, step_scheduler_per_batch=step_scheduler_per_batch,
        )
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss, val_acc = evaluate(model, valloader, criterion, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Step scheduler per epoch if requested
        if scheduler is not None and not step_scheduler_per_batch:
            scheduler.step()

        # Checkpoint
        if checkpoint is not None:
            checkpoint(val_acc, epoch, optimizer, scheduler, scaler)

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time

        # Log epoch
        if logger is not None:
            logger.log_epoch(
                epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_acc=val_acc,
                lr=current_lr,
                epoch_time=elapsed,
            )

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

    # Finalise logger
    if logger is not None:
        best = checkpoint.best_score if checkpoint is not None else None
        logger.finish(best_val_acc=best)

    return train_losses, val_losses, val_accuracies
