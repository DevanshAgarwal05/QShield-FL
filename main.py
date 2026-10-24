import json
import os
from datetime import datetime

import torch

from config import DATA_DIR, DEFAULT_N_QUBITS, DEFAULT_VQC_LAYERS, LABELS, RESULTS_DIR
from data.loader import get_binary_tabular_loaders
from models.hybrid_qnn import HybridQNN
from utils.checkpoints import save_checkpoint
from utils.training import (
    compute_pos_weight_from_loaders,
    run_experiment,
    save_plot,
    save_results,
    set_seed,
)


def save_models(out_dir, models_dict, in_dim, threshold=0.5):
    """Save model weights for inference (the backend loads these)."""
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "in_dim": int(in_dim),
        "threshold": float(threshold),
        "label_names": {str(k): v for k, v in LABELS.items()},
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for name, state_dict in models_dict.items():
        save_checkpoint(
            os.path.join(out_dir, f"model_{name}.pt"),
            state_dict,
            in_dim=in_dim,
            threshold=threshold,
            n_qubits=DEFAULT_N_QUBITS,
            vqc_layers=DEFAULT_VQC_LAYERS,
        )


def main():
    set_seed(42)

    device = "cpu"
    print("Using device:", device)
    print("CPU cores (logical):", os.cpu_count())
    print("Torch threads:", torch.get_num_threads())
    print("-" * 60)

    rounds = 20
    local_epochs = 2
    local_lr = 3e-3
    batch_size = 256

    dataset = "nsl_kdd"
    data_dir = str(DATA_DIR)
    use_20_percent = True

    attack_scale = 5.0
    malicious_client = 0

    n_def = 9
    trim_ratio = 0.25

    dp_sigma = 0.003
    dp_start_round = 8

    out_dir = RESULTS_DIR / "results_full_throttle_100pct"
    out_dir.mkdir(parents=True, exist_ok=True)

    def loaders_for(num_clients):
        return get_binary_tabular_loaders(
            num_clients=num_clients,
            batch_size=batch_size,
            dataset=dataset,
            data_dir=data_dir,
            seed=42,
            split="iid",
            use_20_percent=use_20_percent,
        )

    def scenario(name, loaders, test_loader, in_dim, num_clients, pos_weight, **kwargs):
        return run_experiment(
            name=name,
            model_factory=lambda: HybridQNN(in_dim=in_dim, n_qubits=DEFAULT_N_QUBITS, vqc_layers=DEFAULT_VQC_LAYERS),
            client_loaders=loaders,
            test_loader=test_loader,
            device=device,
            num_clients=num_clients,
            rounds=rounds,
            local_epochs=local_epochs,
            local_lr=local_lr,
            pos_weight=pos_weight,
            **kwargs,
        )

    n1 = 3
    loaders1, test1, in_dim1 = loaders_for(n1)
    pos_w_3 = compute_pos_weight_from_loaders(loaders1)

    acc_clean, state_clean = scenario(
        "Clean FL (FedAvg) [3 clients]", loaders1, test1, in_dim1, n1, pos_w_3,
    )
    acc_attack, state_attack = scenario(
        "Attacked FL (FedAvg + SignFlip) [3 clients]", loaders1, test1, in_dim1, n1, pos_w_3,
        use_attack=True, malicious_client=malicious_client, attack_scale=attack_scale,
    )

    loadersD, testD, in_dimD = loaders_for(n_def)
    pos_w_9 = compute_pos_weight_from_loaders(loadersD)

    acc_def, state_def = scenario(
        "Defended FL (TrimmedMean + SignFlip) [9 clients]", loadersD, testD, in_dimD, n_def, pos_w_9,
        use_attack=True, malicious_client=malicious_client, attack_scale=attack_scale,
        use_defense=True, trim_ratio=trim_ratio,
    )
    acc_def_dp, state_def_dp = scenario(
        "Defended+DP FL (TrimmedMean + DP(warmup) + SignFlip) [9 clients]", loadersD, testD, in_dimD, n_def, pos_w_9,
        use_attack=True, malicious_client=malicious_client, attack_scale=attack_scale,
        use_defense=True, trim_ratio=trim_ratio,
        use_dp=True, dp_sigma=dp_sigma, dp_start_round=dp_start_round,
    )

    curves = {
        "clean": acc_clean,
        "attack": acc_attack,
        "defended": acc_def,
        "defended_dp": acc_def_dp,
    }

    meta_text = f"""HQFL Run
timestamp: {datetime.now().isoformat()}

device: {device}
dataset: {dataset}
data_dir: {data_dir}
use_20_percent: {use_20_percent}

rounds: {rounds}
local_epochs: {local_epochs}
local_lr: {local_lr}
batch_size: {batch_size}

pos_weight_3_clients: {pos_w_3}
pos_weight_9_clients: {pos_w_9}

attack:
  malicious_client: {malicious_client}
  attack_scale: {attack_scale}

defense:
  n_def_clients: {n_def}
  trim_ratio: {trim_ratio}

dp:
  dp_sigma: {dp_sigma}
  dp_start_round: {dp_start_round}
"""

    save_results(out_dir, curves, meta_text, xlsx_name="results_full_throttle.xlsx")
    save_plot(out_dir, curves, "HQFL: Accuracy vs Round", png_name="results_full_throttle.png")

    if in_dim1 == in_dimD:
        save_models(
            out_dir,
            {"clean": state_clean, "attack": state_attack, "defended": state_def, "defended_dp": state_def_dp},
            in_dim=in_dimD,
        )
    else:
        save_models(out_dir, {"clean_3c": state_clean, "attack_3c": state_attack}, in_dim=in_dim1)
        save_models(out_dir, {"defended_9c": state_def, "defended_dp_9c": state_def_dp}, in_dim=in_dimD)

    print("\nDONE. Check folder:", out_dir)


if __name__ == "__main__":
    main()
