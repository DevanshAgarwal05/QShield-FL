import argparse
import json
import os
from datetime import datetime

import torch

from config import DATA_DIR, RESULTS_DIR
from data.loader import get_binary_tabular_loaders
from models.hybrid_qnn import HybridQNN
from utils.checkpoints import save_checkpoint
from utils.svd import apply_svd_to_loaders
from utils.training import (
    compute_pos_weight_from_loaders,
    ensure_unique_dir,
    run_experiment,
    save_plot,
    save_results,
    set_seed,
)


def save_models_svd(out_dir, models_dict, model_in_dim, rank_tag, svd_mean_vec, svd_components,
                    n_qubits, vqc_layers, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "path_type": "svd_hybrid_qnn",
        "svd_enabled": True,
        "svd_rank": int(rank_tag),
        "model_in_dim_after_svd": int(model_in_dim),
        "threshold": float(threshold),
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, f"model_meta_svd_r{rank_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for name, state_dict in models_dict.items():
        save_checkpoint(
            os.path.join(out_dir, f"model_{name}_svd_r{rank_tag}.pt"),
            state_dict,
            in_dim=model_in_dim,
            threshold=threshold,
            n_qubits=n_qubits,
            vqc_layers=vqc_layers,
            svd_mean=svd_mean_vec,
            svd_components=svd_components,
            svd_rank=rank_tag,
        )


def run_rank_experiment(args, rank):
    set_seed(args.seed)
    device = "cpu"

    def loaders_for(num_clients):
        return get_binary_tabular_loaders(
            num_clients=num_clients,
            batch_size=args.batch_size,
            dataset=args.dataset,
            data_dir=args.data_dir,
            seed=args.seed,
            split=args.split,
            use_20_percent=args.use_20_percent,
        )

    def prepare(num_clients):
        loaders, test, in_dim = loaders_for(num_clients)
        if not args.use_svd:
            return loaders, test, in_dim, torch.zeros(in_dim), torch.eye(in_dim)
        return apply_svd_to_loaders(loaders, test, rank=rank, batch_size=args.batch_size)

    loaders3, test3, in_dim3, svd_mean3, svd_comp3 = prepare(3)
    loaders9, test9, in_dim9, svd_mean9, svd_comp9 = prepare(9)

    pos_w_3 = compute_pos_weight_from_loaders(loaders3)
    pos_w_9 = compute_pos_weight_from_loaders(loaders9)

    def scenario(name, loaders, test_loader, in_dim, num_clients, pos_weight, **kwargs):
        return run_experiment(
            name=f"{name}_svd_r{rank}",
            model_factory=lambda: HybridQNN(in_dim=in_dim, n_qubits=args.n_qubits, vqc_layers=args.vqc_layers),
            client_loaders=loaders,
            test_loader=test_loader,
            device=device,
            num_clients=num_clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            local_lr=args.local_lr,
            pos_weight=pos_weight,
            **kwargs,
        )

    attack = dict(use_attack=True, malicious_client=args.malicious_client, attack_scale=args.attack_scale)
    defense = dict(use_defense=True, trim_ratio=args.trim_ratio)

    acc_clean, state_clean = scenario("clean", loaders3, test3, in_dim3, 3, pos_w_3)
    acc_attack, state_attack = scenario("attack", loaders3, test3, in_dim3, 3, pos_w_3, **attack)
    acc_def, state_def = scenario("defended", loaders9, test9, in_dim9, 9, pos_w_9, **attack, **defense)
    acc_def_dp, state_def_dp = scenario(
        "defended_dp", loaders9, test9, in_dim9, 9, pos_w_9, **attack, **defense,
        use_dp=True, dp_sigma=args.dp_sigma, dp_start_round=args.dp_start_round,
    )

    out_dir = ensure_unique_dir(RESULTS_DIR / f"results_svd_hybrid_r{rank}")

    curves = {
        "clean": acc_clean,
        "attack": acc_attack,
        "defended": acc_def,
        "defended_dp": acc_def_dp,
    }
    meta_text = f"""SVD-HQFL Run
timestamp: {datetime.now().isoformat()}
path_type: svd_hybrid_qnn
dataset: {args.dataset}
data_dir: {args.data_dir}
use_20_percent: {args.use_20_percent}
split: {args.split}
device: {device}

svd:
  enabled: {args.use_svd}
  rank: {rank}
  fit_scope: per_experiment_global_train_concat

qnn:
  n_qubits: {args.n_qubits}
  vqc_layers: {args.vqc_layers}

training:
  rounds: {args.rounds}
  local_epochs: {args.local_epochs}
  local_lr: {args.local_lr}
  batch_size: {args.batch_size}

attack:
  malicious_client: {args.malicious_client}
  attack_scale: {args.attack_scale}

defense:
  n_def_clients: 9
  trim_ratio: {args.trim_ratio}

dp:
  dp_sigma: {args.dp_sigma}
  dp_start_round: {args.dp_start_round}
"""
    save_results(out_dir, curves, meta_text, xlsx_name="results_svd_hybrid.xlsx")
    save_plot(out_dir, curves, f"SVD-HQFL (rank={rank}): Accuracy vs Round", png_name="results_svd_hybrid.png")

    # The 9-client projection is saved with every checkpoint (the 3-client one is used when dims differ).
    use9 = in_dim9 == in_dim3
    save_models_svd(
        out_dir,
        {"clean": state_clean, "attack": state_attack, "defended": state_def, "defended_dp": state_def_dp},
        model_in_dim=in_dim9 if use9 else in_dim3,
        rank_tag=rank,
        svd_mean_vec=svd_mean9 if use9 else svd_mean3,
        svd_components=svd_comp9 if use9 else svd_comp3,
        n_qubits=args.n_qubits,
        vqc_layers=args.vqc_layers,
    )

    print(f"[DONE] rank={rank}, out_dir={os.path.abspath(out_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVD-enhanced hybrid classical+quantum FL runner."
    )
    parser.add_argument("--dataset", type=str, default="nsl_kdd")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--use_20_percent", action="store_true")
    parser.add_argument("--split", type=str, default="iid", choices=["iid", "dirichlet"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_svd", action="store_true")
    parser.add_argument("--ranks", type=str, default="24")

    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--local_lr", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--malicious_client", type=int, default=0)
    parser.add_argument("--attack_scale", type=float, default=5.0)
    parser.add_argument("--trim_ratio", type=float, default=0.25)
    parser.add_argument("--dp_sigma", type=float, default=0.003)
    parser.add_argument("--dp_start_round", type=int, default=8)

    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqc_layers", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()
    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    if not ranks:
        raise ValueError("No valid ranks provided in --ranks")

    if not args.use_svd:
        # A no-SVD run still works, using the first rank tag.
        ranks = [ranks[0]]

    for rank in ranks:
        run_rank_experiment(args, rank)


if __name__ == "__main__":
    main()
