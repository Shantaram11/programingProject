from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ts_platform.core.pipeline import TrainingResult


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    path: Path
    display_name: str = ""


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default() -> "RunStore":
        return RunStore(Path.home() / ".ts_model_platform")

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def list_runs(self) -> List[RunMeta]:
        if not self.runs_root.exists():
            return []
        runs: List[RunMeta] = []
        for p in sorted(self.runs_root.iterdir(), reverse=True):
            if p.is_dir():
                display_name = ""
                info_path = p / "info.json"
                if info_path.exists():
                    try:
                        info = json.loads(info_path.read_text(encoding="utf-8"))
                        display_name = str(info.get("display_name") or "")
                    except Exception:
                        display_name = ""
                runs.append(RunMeta(run_id=p.name, path=p, display_name=display_name))
        return runs

    def save_run(self, run_id: str, result: TrainingResult, display_name: str = "") -> Path:
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save info/config/metrics
        info = {
            "run_id": run_id,
            "display_name": display_name,
            "targets": result.targets,
            "models": result.models,
            "metrics_by_model": result.metrics_by_model,
            "train_metrics_by_model": getattr(result, "train_metrics_by_model", {}),
            "config": result.config,
        }
        (run_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save predictions (long format)
        rows: List[Dict[str, Any]] = []
        for target, series in result.per_target.items():
            t = series["t"]
            y_true = np.asarray(series["y_true"], dtype=float)
            preds = series["preds"]
            for i in range(len(t)):
                rec: Dict[str, Any] = {
                    "target": target,
                    "t": str(pd.to_datetime(t[i])),
                    "y_true": float(y_true[i]),
                }
                for model_name, yhat in preds.items():
                    rec[f"pred__{model_name}"] = float(np.asarray(yhat, dtype=float)[i])
                rows.append(rec)
        pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)

        # Save a plot image (one subplot per target)
        self._save_plot_png(run_dir / "plot.png", result)
        return run_dir

    def load_run_info(self, run_id: str) -> Dict[str, Any]:
        p = self.run_dir(run_id) / "info.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing info.json for run: {run_id}")
        return json.loads(p.read_text(encoding="utf-8"))

    def delete_run(self, run_id: str) -> None:
        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def export_run(self, run_id: str, target_dir: Path) -> Path:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        src = self.run_dir(run_id)
        dst = target_dir / run_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return dst

    def _save_plot_png(self, path: Path, result: TrainingResult) -> None:
        import matplotlib.pyplot as plt

        n_targets = max(1, len(result.targets))
        fig_h = 3.6 * n_targets
        fig, axes = plt.subplots(n_targets, 1, figsize=(12, fig_h), sharex=False, tight_layout=True)
        if n_targets == 1:
            axes = [axes]

        for ax, target in zip(axes, result.targets):
            series = result.per_target[target]
            t = pd.to_datetime(series["t"])
            y_true = np.asarray(series["y_true"], dtype=float)
            ax.plot(t, y_true, label="Actual", linewidth=2.2, alpha=0.9)
            for model_name, yhat in series["preds"].items():
                ax.plot(t, np.asarray(yhat, dtype=float), label=model_name, linewidth=1.8, alpha=0.9)
            ax.set_title(f"Target: {target}")
            ax.grid(True, alpha=0.25)
            ax.legend()

        fig.savefig(str(path), dpi=160)
        plt.close(fig)

