import os
import json
import traceback
from datetime import datetime
from copy import deepcopy
from time import perf_counter

import torch
import matplotlib.pyplot as plt
import pandas as pd

from data.loader import get_binary_tabular_loaders
from models.hybrid_qnn import HybridQNN
from federated.client import train_one_client, get_update, apply_update
from federated.aggregation import fedavg, trimmed_mean
from federated.attacks import sign_flip
from federated.privacy import add_dp_noise


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
    neg = (y_all.numel() - pos)
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

    probs_all = torch.cat(probs_all, dim=0) if len(probs_all) else torch.tensor([])
    acc = correct / max(total, 1)

    stats = {
        "mean": float(probs_all.mean().item()) if probs_all.numel() else 0.0,
        "min": float(probs_all.min().item()) if probs_all.numel() else 0.0,
        "max": float(probs_all.max().item()) if probs_all.numel() else 0.0,
    }
    return acc, stats


def run_experiment(
    name,
    client_loaders,
    test_loader,
    in_dim,
    device="cpu",
    num_clients=3,
    rounds=20,
    local_epochs=1,
    local_lr=5e-3,
    use_attack=False,
    malicious_client=0,
    attack_scale=5.0,
    use_dp=False,
    dp_sigma=0.005,
    dp_start_round=6,
    use_defense=False,
    trim_ratio=0.25,
    pos_weight=None,
):
    print(f"\n=== EXPERIMENT: {name} ===")
    print(f"clients={num_clients}, rounds={rounds}, local_epochs={local_epochs}, lr={local_lr}")
    print(f"attack={use_attack}, defense={use_defense}, trim_ratio={trim_ratio}")
    print(f"dp={use_dp}, dp_sigma={dp_sigma}, dp_start_round={dp_start_round}")
    print(f"pos_weight={pos_weight}\n")

    global_model = HybridQNN(in_dim=in_dim, n_qubits=4, vqc_layers=3).to(device)
    global_state = deepcopy(global_model.state_dict())

    client_models = [
        HybridQNN(in_dim=in_dim, n_qubits=4, vqc_layers=3).to(device)
        for _ in range(num_clients)
    ]

    acc_curve = []

    try:
        for r in range(1, rounds + 1):
            round_t0 = perf_counter()
            print(f"\nStarting Round {r}/{rounds}")

            updates = []

            for cid in range(num_clients):
                t0 = perf_counter()

                # sync client with global state
                client_models[cid].load_state_dict(global_state)

                # ✅ FIXED: your function expects dataloader= not loader=
                train_one_client(
                    model=client_models[cid],
                    dataloader=client_loaders[cid],
                    epochs=local_epochs,
                    lr=local_lr,
                    device=device,
                    pos_weight=pos_weight,
                )

                # compute update
                update = get_update(global_state, client_models[cid].state_dict())

                # attack (optional)
                if use_attack and cid == malicious_client:
                    update = sign_flip(update, scale=attack_scale)
                    print(f"  [ATTACK] Client {cid} performed SignFlip (scale={attack_scale})")

                # dp (optional, after warmup)
                if use_dp and r >= dp_start_round:
                    update = add_dp_noise(update, sigma=dp_sigma)
                    print(f"  [DP] Added noise at round {r} (sigma={dp_sigma})")

                updates.append(update)

                t1 = perf_counter()
                print(f"  Client {cid} done in {(t1 - t0):.2f}s")

            # aggregate
            if use_defense:
                agg_update = trimmed_mean(updates, trim_ratio=trim_ratio)
            else:
                agg_update = fedavg(updates)

            # apply update to global
            global_state = apply_update(global_state, agg_update)
            global_model.load_state_dict(global_state)

            # evaluate
            acc, stats = evaluate_with_stats(global_model, test_loader, device=device)
            acc_curve.append(acc)

            round_t1 = perf_counter()
            print(f"\nRound {r} complete in {(round_t1 - round_t0):.2f}s")
            print(f"  Test Acc: {acc:.4f}")
            print(f"  Prob(mean/min/max): {stats['mean']:.4f}/{stats['min']:.4f}/{stats['max']:.4f}")
            print("-" * 60)

    except Exception:
        print(f"\n[run_experiment] FAILED: {name}")
        traceback.print_exc()

    return acc_curve, deepcopy(global_state)


def save_results(out_dir, curves_dict, meta_text):
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[save_results] Saving into: {os.path.abspath(out_dir)}")
        print(f"[save_results] CWD: {os.getcwd()}")

        wrote_any_csv = False
        for name, curve in curves_dict.items():
            if curve is None or len(curve) == 0:
                continue
            rounds_axis = list(range(1, len(curve) + 1))
            df = pd.DataFrame({"round": rounds_axis, "accuracy": curve})
            path = os.path.join(out_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            wrote_any_csv = True
            print(f"[save_results] Wrote CSV: {os.path.abspath(path)}")

        meta_path = os.path.join(out_dir, "meta.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(meta_text)
        print(f"[save_results] Wrote Meta: {os.path.abspath(meta_path)}")

        # ✅ Only write Excel if we have at least one non-empty curve
        if wrote_any_csv:
            try:
                xlsx_path = os.path.join(out_dir, "results_full_throttle.xlsx")
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    for name, curve in curves_dict.items():
                        if curve is None or len(curve) == 0:
                            continue
                        rounds_axis = list(range(1, len(curve) + 1))
                        df = pd.DataFrame({"round": rounds_axis, "accuracy": curve})
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                print(f"[save_results] Wrote Excel: {os.path.abspath(xlsx_path)}")
            except Exception:
                print("[save_results] Excel write failed (CSV still saved). Traceback:")
                traceback.print_exc()
        else:
            print("[save_results] Skipped Excel (no curves to write).")

        print(f"[save_results] Files now in folder: {os.listdir(out_dir)}\n")

    except Exception:
        print("[save_results] FAILED. Traceback:")
        traceback.print_exc()


def save_models(out_dir, models_dict, in_dim, threshold=0.5):
    """
    Saves model weights for inference (backend can load these).
    """
    try:
        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "in_dim": int(in_dim),
            "threshold": float(threshold),
            "label_names": {
                "0": "Class 0",
                "1": "Class 1"
            },
            "saved_at": datetime.now().isoformat()
        }
        meta_path = os.path.join(out_dir, "model_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save_models] Wrote: {os.path.abspath(meta_path)}")

        for name, state_dict in models_dict.items():
            if state_dict is None:
                continue
            path = os.path.join(out_dir, f"model_{name}.pt")
            torch.save(
                {
                    "state_dict": state_dict,
                    "in_dim": int(in_dim),
                    "threshold": float(threshold),
                },
                path
            )
            print(f"[save_models] Wrote model: {os.path.abspath(path)}")

    except Exception:
        print("[save_models] FAILED. Traceback:")
        traceback.print_exc()


def plot_curves(out_dir, curves_dict, title):
    try:
        # ✅ Only plot if at least one curve exists
        any_curve = any(curve is not None and len(curve) > 0 for curve in curves_dict.values())
        if not any_curve:
            print("[plot_curves] Skipped plot (no curves).")
            return

        plt.figure()
        plotted_any = False

        for name, curve in curves_dict.items():
            if curve is None or len(curve) == 0:
                continue
            rounds_axis = list(range(1, len(curve) + 1))
            plt.plot(rounds_axis, curve, marker="o", label=name)
            plotted_any = True

        plt.xlabel("Round")
        plt.ylabel("Test Accuracy")
        plt.title(title)
        if plotted_any:
            plt.legend()
        plt.grid(True)

        plot_path = os.path.join(out_dir, "results_full_throttle.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nSaved plot to: {plot_path}")

    except Exception:
        print("[plot_curves] Plotting failed. Traceback:")
        traceback.print_exc()


def main():
    set_seed(42)

    device = "cpu"
    print("Using device:", device)
    print("CPU cores (logical):", os.cpu_count())
    print("Torch threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())
    print("-" * 60)

    rounds = 20
    local_epochs = 2
    local_lr = 3e-3
    batch_size = 256

    dataset = "nsl_kdd"
    data_dir = "data/nsl_kdd"
    use_20_percent = True

    attack_scale = 5.0
    malicious_client = 0

    n_def = 9
    trim_ratio = 0.25

    dp_sigma = 0.003
    dp_start_round = 8

    out_dir = "results_full_throttle_100pct"
    os.makedirs(out_dir, exist_ok=True)

    # 3-client run
    n1 = 3
    loaders1, test_loader1, in_dim1 = get_binary_tabular_loaders(
        num_clients=n1,
        batch_size=batch_size,
        dataset=dataset,
        data_dir=data_dir,
        seed=42,
        split="iid",
        use_20_percent=use_20_percent
    )
    pos_w_3 = compute_pos_weight_from_loaders(loaders1)

    (acc_clean, state_clean) = run_experiment(
        name="Clean FL (FedAvg) [3 clients]",
        client_loaders=loaders1,
        test_loader=test_loader1,
        in_dim=in_dim1,
        device=device,
        num_clients=n1,
        rounds=rounds,
        local_epochs=local_epochs,
        local_lr=local_lr,
        use_attack=False,
        use_defense=False,
        use_dp=False,
        pos_weight=pos_w_3,
    )

    (acc_attack, state_attack) = run_experiment(
        name="Attacked FL (FedAvg + SignFlip) [3 clients]",
        client_loaders=loaders1,
        test_loader=test_loader1,
        in_dim=in_dim1,
        device=device,
        num_clients=n1,
        rounds=rounds,
        local_epochs=local_epochs,
        local_lr=local_lr,
        use_attack=True,
        malicious_client=malicious_client,
        attack_scale=attack_scale,
        use_defense=False,
        use_dp=False,
        pos_weight=pos_w_3,
    )

    # 9-client defended run
    loadersD, test_loaderD, in_dimD = get_binary_tabular_loaders(
        num_clients=n_def,
        batch_size=batch_size,
        dataset=dataset,
        data_dir=data_dir,
        seed=42,
        split="iid",
        use_20_percent=use_20_percent
    )
    pos_w_9 = compute_pos_weight_from_loaders(loadersD)

    (acc_def_only, state_def_only) = run_experiment(
        name="Defended FL (TrimmedMean + SignFlip) [9 clients]",
        client_loaders=loadersD,
        test_loader=test_loaderD,
        in_dim=in_dimD,
        device=device,
        num_clients=n_def,
        rounds=rounds,
        local_epochs=local_epochs,
        local_lr=local_lr,
        use_attack=True,
        malicious_client=malicious_client,
        attack_scale=attack_scale,
        use_defense=True,
        trim_ratio=trim_ratio,
        use_dp=False,
        pos_weight=pos_w_9,
    )

    (acc_def_dp, state_def_dp) = run_experiment(
        name="Defended+DP FL (TrimmedMean + DP(warmup) + SignFlip) [9 clients]",
        client_loaders=loadersD,
        test_loader=test_loaderD,
        in_dim=in_dimD,
        device=device,
        num_clients=n_def,
        rounds=rounds,
        local_epochs=local_epochs,
        local_lr=local_lr,
        use_attack=True,
        malicious_client=malicious_client,
        attack_scale=attack_scale,
        use_defense=True,
        trim_ratio=trim_ratio,
        use_dp=True,
        dp_sigma=dp_sigma,
        dp_start_round=dp_start_round,
        pos_weight=pos_w_9,
    )

    curves = {
        "clean": acc_clean,
        "attack": acc_attack,
        "defended": acc_def_only,
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

    save_results(out_dir, curves, meta_text)
    plot_curves(out_dir, curves, "HQFL: Accuracy vs Round")

    # Save trained models for inference
    if in_dim1 == in_dimD:
        save_models(
            out_dir=out_dir,
            models_dict={
                "clean": state_clean,
                "attack": state_attack,
                "defended": state_def_only,
                "defended_dp": state_def_dp,
            },
            in_dim=in_dimD,
            threshold=0.5
        )
    else:
        save_models(
            out_dir=out_dir,
            models_dict={
                "clean_3c": state_clean,
                "attack_3c": state_attack,
            },
            in_dim=in_dim1,
            threshold=0.5
        )
        save_models(
            out_dir=out_dir,
            models_dict={
                "defended_9c": state_def_only,
                "defended_dp_9c": state_def_dp,
            },
            in_dim=in_dimD,
            threshold=0.5
        )

    print("\nDONE. Check folder:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
