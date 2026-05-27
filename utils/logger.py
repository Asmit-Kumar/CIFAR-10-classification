"""
RunLogger — Lightweight experiment tracker for CIFAR-10 / CIFAR-100.

Each run is saved as its own JSON file inside a log directory:

    logs/ImageClassifierTorch_ResNet/
        resnet34_20260521_100737.json
        resnet34_20260521_153022.json   ← never overwritten

Usage (via fit()):
    fit(..., log=True)   # everything is automatic

Usage (standalone):
    logger = RunLogger(log_dir="logs/my_experiment")
    logger.start(config={"model": "ResNet34", "lr": 3e-4})
    for epoch in range(epochs):
        logger.log_epoch(epoch, train_loss=tl, val_loss=vl, val_acc=va, lr=lr)
    logger.finish()
    logger.summary()
"""

import json
import time
from datetime import datetime
from pathlib import Path


class RunLogger:
    """
    Lightweight experiment logger — one JSON file per run, never overwritten.

    Args:
        run_name (str | None): Human-readable name for this run.  When ``None``
            (default), a name is auto-generated in ``start()`` as
            ``{model}_{YYYYMMDD}_{HHMMSS}``.
        log_dir (str): Directory in which run files are saved.
            Each run creates ``{log_dir}/{run_name}.json``.
        verbose (bool): Print a one-line summary after each epoch.
    """

    def __init__(
        self,
        run_name: str | None = None,
        log_dir: str = "logs/runs",
        verbose: bool = True,
    ):
        self._run_name_override = run_name   # None → auto-generate in start()
        self.run_name = run_name or "(pending)"
        
        # Centralize relative logs under project root
        log_path = Path(log_dir)
        if not log_path.is_absolute() and log_path.parts and log_path.parts[0] == "logs":
            project_root = Path(__file__).resolve().parent.parent
            self.log_dir = project_root / log_path
        else:
            self.log_dir = log_path

        self.verbose = verbose

        # Current run state
        self._run: dict = {}
        self._epoch_metrics: list[dict] = []
        self._run_start: float = 0.0

        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, config: dict | None = None) -> None:
        """
        Begin a new run. Call this once before your training loop.

        Args:
            config: Arbitrary dict of hyperparameters / metadata to store
                    alongside the run (e.g. model name, LR, batch size, …).

        Run naming:
            If ``run_name`` was not passed to ``__init__``, a unique name is
            generated as ``{model}_{YYYYMMDD}_{HHMMSS}`` using the ``'model'``
            key from *config* (falls back to ``'run'`` if absent).
        """
        self._epoch_metrics = []
        self._run_start = time.time()
        cfg = config or {}

        # Auto-generate a unique, timestamped name
        if self._run_name_override is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_tag = cfg.get("model", "run")
            model_tag = model_tag.lower().replace(" ", "_").replace("/", "_")
            self.run_name = f"{model_tag}_{ts}"

        self._run = {
            "run_name":       self.run_name,
            "started_at":     datetime.now().isoformat(timespec="seconds"),
            "config":         cfg,
            "epochs":         [],    # filled by log_epoch
            "best_val_acc":   None,  # filled by finish
            "total_time_min": None,  # filled by finish
            "status":         "running",
        }

        if self.verbose:
            print(f"[RunLogger] ▶  Run '{self.run_name}' started — "
                  f"saving to '{self._run_file}'")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float | None = None,
        **extra,
    ) -> None:
        """
        Record metrics for a single epoch.

        Args:
            epoch:      0-based epoch index.
            train_loss: Average training loss for this epoch.
            val_loss:   Average validation loss for this epoch.
            val_acc:    Validation accuracy (%).
            lr:         Current learning rate (last param group).
            epoch_time: Wall-clock seconds for the epoch.
            **extra:    Any additional scalar metrics to store.
        """
        entry = {
            "epoch":      epoch + 1,   # 1-based for readability
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 4),
            "lr":         round(lr, 8),
            "epoch_time": round(epoch_time, 2) if epoch_time is not None else None,
        }
        entry.update({k: round(v, 6) if isinstance(v, float) else v
                      for k, v in extra.items()})
        self._epoch_metrics.append(entry)

        if self.verbose:
            t_str = f"  ⏱ {epoch_time:.1f}s" if epoch_time else ""
            print(
                f"[RunLogger] Epoch {epoch + 1:3d} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.2f}%  lr={lr:.2e}{t_str}"
            )

    def finish(self, best_val_acc: float | None = None) -> None:
        """
        Finalise the run and write it to its own JSON file.

        Args:
            best_val_acc: Best validation accuracy (pulled from ModelCheckpoint
                          when called via ``fit()``; computed from epochs if None).
        """
        total_time = (time.time() - self._run_start) / 60.0

        if best_val_acc is None and self._epoch_metrics:
            best_val_acc = max(e["val_acc"] for e in self._epoch_metrics)

        self._run.update({
            "epochs":         self._epoch_metrics,
            "best_val_acc":   round(best_val_acc, 4) if best_val_acc else None,
            "total_time_min": round(total_time, 2),
            "status":         "finished",
            "finished_at":    datetime.now().isoformat(timespec="seconds"),
        })

        self._save_run()

        if self.verbose:
            print(
                f"[RunLogger] ■  Run '{self.run_name}' finished — "
                f"{len(self._epoch_metrics)} epochs in {total_time:.1f} min  |  "
                f"best val acc: {best_val_acc:.2f}%"
            )

    def summary(self, top_n: int = 10) -> None:
        """
        Print a ranked comparison table of all finished runs in ``log_dir``.

        Args:
            top_n: Maximum number of runs to display (sorted by best val acc).
        """
        runs = self._load_all_runs()
        if not runs:
            print(f"[RunLogger] No runs found in '{self.log_dir}'.")
            return

        finished = [r for r in runs if r.get("status") == "finished"]
        finished.sort(key=lambda r: r.get("best_val_acc") or 0, reverse=True)
        display = finished[:top_n]

        col_w = [28, 10, 12, 10, 7]
        headers = ["Run Name", "Best Acc", "Time (min)", "Epochs", "LR"]
        sep = "─" * (sum(col_w) + len(headers) * 2)
        print(f"\n{'Run Summary':^{len(sep)}}")
        print(sep)
        print("  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
        print(sep)

        for r in display:
            cfg = r.get("config", {})
            n_epochs = len(r.get("epochs", []))
            lr = cfg.get("lr", cfg.get("learning_rate", "—"))
            lr_str = f"{lr:.0e}" if isinstance(lr, float) else str(lr)
            vals = [
                r.get("run_name", "—")[:col_w[0]],
                f"{r.get('best_val_acc', 0):.2f}%",
                f"{r.get('total_time_min', 0):.1f}",
                str(n_epochs),
                lr_str,
            ]
            print("  ".join(v.ljust(w) for v, w in zip(vals, col_w)))

        print(sep)
        print(f"  Showing {len(display)} of {len(finished)} finished run(s)  |  "
              f"Dir: {self.log_dir}\n")

    def load_run(self, run_name: str) -> dict | None:
        """
        Load a specific run by name.

        Args:
            run_name: Run name to look up (matches the JSON filename stem).

        Returns:
            The run dict, or None if no matching file is found.
        """
        run_file = self.log_dir / f"{run_name}.json"
        if not run_file.exists():
            return None
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def get_epoch_series(self, run_name: str | None = None) -> dict[str, list]:
        """
        Extract per-epoch metric lists from a run — ready for plotting.

        Args:
            run_name: Run to load. Defaults to the current in-progress run.

        Returns:
            Dict with keys: 'train_loss', 'val_loss', 'val_acc', 'lr', 'epoch_time'.
        """
        if run_name is None:
            epochs = self._epoch_metrics
        else:
            run = self.load_run(run_name)
            if run is None:
                raise KeyError(f"Run '{run_name}' not found in '{self.log_dir}'")
            epochs = run.get("epochs", [])

        keys = ["train_loss", "val_loss", "val_acc", "lr", "epoch_time"]
        return {k: [e.get(k) for e in epochs] for k in keys}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _run_file(self) -> Path:
        """Path to this run's dedicated JSON file."""
        return self.log_dir / f"{self.run_name}.json"

    def _load_all_runs(self) -> list[dict]:
        """Read every *.json file in log_dir and return as a list of run dicts."""
        if not self.log_dir.exists():
            return []
        runs = []
        for path in sorted(self.log_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Support both single-run dicts and legacy lists
                if isinstance(data, list):
                    runs.extend(data)
                else:
                    runs.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return runs

    def _save_run(self) -> None:
        """Write this run to its own dedicated JSON file."""
        with open(self._run_file, "w", encoding="utf-8") as f:
            json.dump(self._run, f, indent=2)
