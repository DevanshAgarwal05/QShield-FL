from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config import DATA_DIR, ROOT
from data.loader import get_binary_tabular_loaders
from models.classical_baselines import BaselineSpec, get_all_baselines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lightweight classical baselines for QSHIELD-FL comparison."
    )
    parser.add_argument("--dataset", type=str, default="nsl_kdd", choices=["nsl_kdd", "breast_cancer"])
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--split", type=str, default="iid", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--use_20_percent", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite default output folder names instead of timestamped variants.",
    )
    return parser.parse_args()


def flatten_client_loaders(client_loaders) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for dl in client_loaders:
        x, y = dl.dataset.tensors
        xs.append(x.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    x_train = np.concatenate(xs, axis=0).astype(np.float32)
    y_train = np.concatenate(ys, axis=0).astype(np.int64)
    return x_train, y_train


def flatten_test_loader(test_loader) -> Tuple[np.ndarray, np.ndarray]:
    x, y = test_loader.dataset.tensors
    x_test = x.detach().cpu().numpy().astype(np.float32)
    y_test = y.detach().cpu().numpy().astype(np.int64)
    return x_test, y_test


def build_output_dir(base_name: str, overwrite: bool) -> Path:
    out = ROOT / base_name
    if overwrite or not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out2 = ROOT / f"{base_name}_{stamp}"
    out2.mkdir(parents=True, exist_ok=True)
    return out2


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Note: project loader maps "normal" -> 1, "attack" -> 0.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def save_outputs(
    out_dir: Path,
    spec: BaselineSpec,
    metrics: Dict[str, float],
    cm: np.ndarray,
    args,
    in_dim: int,
):
    # 1) Summary JSON
    summary = {
        "model": spec.name,
        "description": spec.description,
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "split": args.split,
        "dirichlet_alpha": args.dirichlet_alpha,
        "use_20_percent": bool(args.use_20_percent),
        "in_dim": int(in_dim),
        "label_mapping_from_loader": {
            "0": "attack",
            "1": "normal",
        },
        "metrics": metrics,
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 2) Metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    # 3) Confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=["true_0_attack", "true_1_normal"],
        columns=["pred_0_attack", "pred_1_normal"],
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    # 4) Meta text (project-style quick read)
    meta_txt = (
        f"Classical Baseline Run\n"
        f"timestamp: {summary['timestamp']}\n"
        f"model: {spec.name}\n"
        f"description: {spec.description}\n"
        f"dataset: {args.dataset}\n"
        f"data_dir: {args.data_dir}\n"
        f"num_clients: {args.num_clients}\n"
        f"split: {args.split}\n"
        f"dirichlet_alpha: {args.dirichlet_alpha}\n"
        f"use_20_percent: {bool(args.use_20_percent)}\n"
        f"in_dim: {in_dim}\n"
        f"\n"
        f"metrics:\n"
        f"  accuracy: {metrics['accuracy']:.6f}\n"
        f"  precision: {metrics['precision']:.6f}\n"
        f"  recall: {metrics['recall']:.6f}\n"
        f"  f1: {metrics['f1']:.6f}\n"
        f"\n"
        f"confusion_matrix_labels: [0, 1]\n"
        f"confusion_matrix:\n"
        f"  [[{int(cm[0,0])}, {int(cm[0,1])}],\n"
        f"   [{int(cm[1,0])}, {int(cm[1,1])}]]\n"
    )
    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(meta_txt)


def run_single_baseline(spec: BaselineSpec, x_train, y_train, x_test, y_test, args, in_dim: int):
    print(f"\n=== Running baseline: {spec.name} ===")
    est = spec.estimator
    est.fit(x_train, y_train)
    y_pred = est.predict(x_test).astype(np.int64)

    metrics = evaluate_binary(y_true=y_test, y_pred=y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    out_name = f"results_classical_{spec.name}"
    out_dir = build_output_dir(base_name=out_name, overwrite=args.overwrite)
    save_outputs(out_dir=out_dir, spec=spec, metrics=metrics, cm=cm, args=args, in_dim=in_dim)

    print(f"saved: {out_dir}")
    print(
        f"acc={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
    )


def main():
    args = parse_args()
    client_loaders, test_loader, in_dim = get_binary_tabular_loaders(
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_dir=args.data_dir,
        seed=args.seed,
        split=args.split,
        dirichlet_alpha=args.dirichlet_alpha,
        use_20_percent=args.use_20_percent,
    )

    x_train, y_train = flatten_client_loaders(client_loaders)
    x_test, y_test = flatten_test_loader(test_loader)

    baselines = get_all_baselines(seed=args.seed)
    for spec in baselines:
        run_single_baseline(
            spec=spec,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            args=args,
            in_dim=in_dim,
        )

    print("\nDONE: classical baseline outputs generated.")


if __name__ == "__main__":
    main()
