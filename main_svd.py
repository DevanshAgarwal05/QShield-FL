import argparse
import json
import os
import traceback
from copy import deepcopy
from datetime import datetime
from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.loader import get_binary_tabular_loaders
from federated.aggregation import fedavg, trimmed_mean
from federated.attacks import sign_flip
from federated.client import apply_update, get_update, train_one_client
from federated.privacy import add_dp_noise
from models.hybrid_qnn import HybridQNN

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight_from_loaders(client_loaders):
    ys = []
    for dl in client_loaders:
        y = dl.dataset.tensors[1].detach().cpu().float()
        ys.append(y)
    y_all = torch.cat(ys, dim=0)
    pos = y_all.sum().item()
    neg = y_all.numel() - pos
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def evaluate_with_stats(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    probs_all = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            probs_all.append(probs.detach().cpu())

    probs_all = torch.cat(probs_all, dim=0) if probs_all else torch.tensor([])
    acc = correct / max(total, 1)
    stats = {
        "mean": float(probs_all.mean().item()) if probs_all.numel() else 0.0,
        "min": float(probs_all.min().item()) if probs_all.numel() else 0.0,
        "max": float(probs_all.max().item()) if probs_all.numel() else 0.0,
    }
    return acc, stats


def ensure_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        return base_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{base_dir}_{stamp}"
    i = 1
    while os.path.exists(candidate):
        i += 1
        candidate = f"{base_dir}_{stamp}_{i}"
    return candidate


def fit_svd_projection(client_loaders, rank):
    xs = []
    for dl in client_loaders:
        xs.append(dl.dataset.tensors[0].detach().cpu().float())
    x_all = torch.cat(xs, dim=0)
    d = x_all.shape[1]
    rank = int(max(1, min(rank, d)))

    mean_vec = x_all.mean(dim=0, keepdim=True)
    x_centered = x_all - mean_vec
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    # vh: [d, d], keep first `rank` right singular vectors
    components = vh[:rank, :].T.contiguous()  # [d, rank]

    return mean_vec.squeeze(0), components


def _project_tensor(x, mean_vec, components):
    return (x - mean_vec) @ components


def apply_svd_to_loaders(client_loaders, test_loader, rank, batch_size):
    mean_vec, components = fit_svd_projection(client_loaders, rank)
    mean_vec = mean_vec.float()
    components = components.float()

    new_client_loaders = []
    for dl in client_loaders:
        x, y = dl.dataset.tensors
        x_proj = _project_tensor(x.float(), mean_vec, components)
        ds = TensorDataset(x_proj, y.float())
        new_client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    x_test, y_test = test_loader.dataset.tensors
    x_test_proj = _project_tensor(x_test.float(), mean_vec, components)
    new_test_loader = DataLoader(
        TensorDataset(x_test_proj, y_test.float()),
        batch_size=batch_size,
        shuffle=False,
    )

    return new_client_loaders, new_test_loader, int(components.shape[1]), mean_vec, components


def run_experiment(
    name,
    client_loaders,
    test_loader,
    in_dim,
    device="cpu",
    num_clients=3,
    rounds=20,
    local_epochs=2,
    local_lr=3e-3,
    use_attack=False,
    malicious_client=0,
    attack_scale=5.0,
    use_dp=False,
    dp_sigma=0.003,
    dp_start_round=8,
    use_defense=False,
    trim_ratio=0.25,
    pos_weight=None,
    n_qubits=4,
    vqc_layers=3,
):
    print(f"\n=== EXPERIMENT: {name} ===")
    print(
        f"clients={num_clients}, rounds={rounds}, local_epochs={local_epochs}, "
        f"lr={local_lr}, attack={use_attack}, defense={use_defense}, dp={use_dp}"
    )

    global_model = HybridQNN(
        in_dim=in_dim,
        n_qubits=n_qubits,
        vqc_layers=vqc_layers,
    ).to(device)
    global_state = deepcopy(global_model.state_dict())
    client_models = [
        HybridQNN(in_dim=in_dim, n_qubits=n_qubits, vqc_layers=vqc_layers).to(device)
        for _ in range(num_clients)
    ]

    acc_curve = []
    try:
        for r in range(1, rounds + 1):
            round_t0 = perf_counter()
            updates = []

            for cid in range(num_clients):
                client_models[cid].load_state_dict(global_state)
                train_one_client(
                    model=client_models[cid],
                    dataloader=client_loaders[cid],
                    epochs=local_epochs,
                    lr=local_lr,
                    device=device,
                    pos_weight=pos_weight,
                )

                update = get_update(global_state, client_models[cid].state_dict())
                if use_attack and cid == malicious_client:
                    update = sign_flip(update, scale=attack_scale)
                if use_dp and r >= dp_start_round:
                    update = add_dp_noise(update, sigma=dp_sigma)
                updates.append(update)

            agg_update = trimmed_mean(updates, trim_ratio=trim_ratio) if use_defense else fedavg(updates)
            global_state = apply_update(global_state, agg_update)
            global_model.load_state_dict(global_state)

            acc, stats = evaluate_with_stats(global_model, test_loader, device=device)
            acc_curve.append(acc)
            print(
                f"[{name}] round={r}/{rounds}, acc={acc:.4f}, "
                f"prob(mean/min/max)={stats['mean']:.4f}/{stats['min']:.4f}/{stats['max']:.4f}, "
                f"time={(perf_counter() - round_t0):.2f}s"
            )
    except Exception:
        print(f"[run_experiment] FAILED: {name}")
        traceback.print_exc()

    return acc_curve, deepcopy(global_state)


def save_results(out_dir, curves_dict, meta_text):
    os.makedirs(out_dir, exist_ok=True)
    wrote_any_csv = False

    for name, curve in curves_dict.items():
        if not curve:
            continue
        rounds_axis = list(range(1, len(curve) + 1))
        df = pd.DataFrame({"round": rounds_axis, "accuracy": curve})
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        wrote_any_csv = True

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(meta_text)

    if wrote_any_csv:
        xlsx_path = os.path.join(out_dir, "results_svd_hybrid.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, curve in curves_dict.items():
                if not curve:
                    continue
                rounds_axis = list(range(1, len(curve) + 1))
                df = pd.DataFrame({"round": rounds_axis, "accuracy": curve})
                df.to_excel(writer, sheet_name=name[:31], index=False)


def save_plot(out_dir, curves_dict, title):
    if plt is None:
        print("[save_plot] matplotlib not available, skipping plot generation.")
        return

    any_curve = any(curve is not None and len(curve) > 0 for curve in curves_dict.values())
    if not any_curve:
        return

    plt.figure()
    for name, curve in curves_dict.items():
        if not curve:
            continue
        rounds_axis = list(range(1, len(curve) + 1))
        plt.plot(rounds_axis, curve, marker="o", label=name)
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "results_svd_hybrid.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_models_svd(
    out_dir,
    models_dict,
    model_in_dim,
    rank_tag,
    svd_mean_vec,
    svd_components,
    threshold=0.5,
):
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "path_type": "svd_hybrid_qnn",
        "svd_enabled": True,
        "svd_rank": int(rank_tag),
        "model_in_dim_after_svd": int(model_in_dim),
        "threshold": float(threshold),
        "saved_at": datetime.now().isoformat(),
    }
    with open(
        os.path.join(out_dir, f"model_meta_svd_r{rank_tag}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2)

    for name, state_dict in models_dict.items():
        if state_dict is None:
            continue
        path = os.path.join(out_dir, f"model_{name}_svd_r{rank_tag}.pt")
        torch.save(
            {
                "state_dict": state_dict,
                "in_dim": int(model_in_dim),
                "threshold": float(threshold),
                "svd_enabled": True,
                "svd_rank": int(rank_tag),
                "svd_mean": svd_mean_vec.cpu(),
                "svd_components": svd_components.cpu(),
            },
            path,
        )


def run_rank_experiment(args, rank):
    set_seed(args.seed)
    device = "cpu"

    loaders3, test3, in_dim3 = get_binary_tabular_loaders(
        num_clients=3,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_dir=args.data_dir,
        seed=args.seed,
        split=args.split,
        use_20_percent=args.use_20_percent,
    )
    loaders9, test9, in_dim9 = get_binary_tabular_loaders(
        num_clients=9,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_dir=args.data_dir,
        seed=args.seed,
        split=args.split,
        use_20_percent=args.use_20_percent,
    )

    if not args.use_svd:
        svd_mean3 = torch.zeros(in_dim3)
        svd_comp3 = torch.eye(in_dim3)
        svd_mean9 = torch.zeros(in_dim9)
        svd_comp9 = torch.eye(in_dim9)
        in_dim3_eff, in_dim9_eff = in_dim3, in_dim9
    else:
        loaders3, test3, in_dim3_eff, svd_mean3, svd_comp3 = apply_svd_to_loaders(
            loaders3, test3, rank=rank, batch_size=args.batch_size
        )
        loaders9, test9, in_dim9_eff, svd_mean9, svd_comp9 = apply_svd_to_loaders(
            loaders9, test9, rank=rank, batch_size=args.batch_size
        )

    pos_w_3 = compute_pos_weight_from_loaders(loaders3)
    pos_w_9 = compute_pos_weight_from_loaders(loaders9)

    acc_clean, state_clean = run_experiment(
        name=f"clean_svd_r{rank}",
        client_loaders=loaders3,
        test_loader=test3,
        in_dim=in_dim3_eff,
        device=device,
        num_clients=3,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        use_attack=False,
        use_defense=False,
        use_dp=False,
        pos_weight=pos_w_3,
        n_qubits=args.n_qubits,
        vqc_layers=args.vqc_layers,
    )
    acc_attack, state_attack = run_experiment(
        name=f"attack_svd_r{rank}",
        client_loaders=loaders3,
        test_loader=test3,
        in_dim=in_dim3_eff,
        device=device,
        num_clients=3,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        use_attack=True,
        malicious_client=args.malicious_client,
        attack_scale=args.attack_scale,
        use_defense=False,
        use_dp=False,
        pos_weight=pos_w_3,
        n_qubits=args.n_qubits,
        vqc_layers=args.vqc_layers,
    )
    acc_def, state_def = run_experiment(
        name=f"defended_svd_r{rank}",
        client_loaders=loaders9,
        test_loader=test9,
        in_dim=in_dim9_eff,
        device=device,
        num_clients=9,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        use_attack=True,
        malicious_client=args.malicious_client,
        attack_scale=args.attack_scale,
        use_defense=True,
        trim_ratio=args.trim_ratio,
        use_dp=False,
        pos_weight=pos_w_9,
        n_qubits=args.n_qubits,
        vqc_layers=args.vqc_layers,
    )
    acc_def_dp, state_def_dp = run_experiment(
        name=f"defended_dp_svd_r{rank}",
        client_loaders=loaders9,
        test_loader=test9,
        in_dim=in_dim9_eff,
        device=device,
        num_clients=9,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        use_attack=True,
        malicious_client=args.malicious_client,
        attack_scale=args.attack_scale,
        use_defense=True,
        trim_ratio=args.trim_ratio,
        use_dp=True,
        dp_sigma=args.dp_sigma,
        dp_start_round=args.dp_start_round,
        pos_weight=pos_w_9,
        n_qubits=args.n_qubits,
        vqc_layers=args.vqc_layers,
    )

    base_out = f"results_svd_hybrid_r{rank}"
    out_dir = ensure_unique_dir(base_out)
    os.makedirs(out_dir, exist_ok=True)

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
device: cpu

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
    save_results(out_dir, curves, meta_text)
    save_plot(out_dir, curves, f"SVD-HQFL (rank={rank}): Accuracy vs Round")

    # Save the 9-client projection; this is the path used for defended variants.
    save_models_svd(
        out_dir=out_dir,
        models_dict={
            "clean": state_clean,
            "attack": state_attack,
            "defended": state_def,
            "defended_dp": state_def_dp,
        },
        model_in_dim=in_dim9_eff if in_dim9_eff == in_dim3_eff else in_dim3_eff,
        rank_tag=rank,
        svd_mean_vec=svd_mean9 if in_dim9_eff == in_dim3_eff else svd_mean3,
        svd_components=svd_comp9 if in_dim9_eff == in_dim3_eff else svd_comp3,
        threshold=0.5,
    )

    print(f"[DONE] rank={rank}, out_dir={os.path.abspath(out_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel SVD-enhanced hybrid classical+quantum FL runner."
    )
    parser.add_argument("--dataset", type=str, default="nsl_kdd")
    parser.add_argument("--data_dir", type=str, default="data/nsl_kdd")
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
        # Keep explicit behavior: a no-SVD run still works, using the first rank tag.
        ranks = [ranks[0]]

    for rank in ranks:
        run_rank_experiment(args, rank)


if __name__ == "__main__":
    main()
