"""Training/evaluation loop and result writers shared by the main*.py entry points."""
import os
from copy import deepcopy
from datetime import datetime
from time import perf_counter

import pandas as pd
import torch

from federated.aggregation import fedavg, trimmed_mean
from federated.attacks import sign_flip
from federated.client import apply_update, get_update, train_one_client
from federated.privacy import add_dp_noise


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight_from_loaders(client_loaders):
    y_all = torch.cat([dl.dataset.tensors[1].detach().cpu().float() for dl in client_loaders], dim=0)
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
            probs = torch.sigmoid(model(x))
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


def run_experiment(
    name,
    model_factory,
    client_loaders,
    test_loader,
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
):
    """Run one federated scenario and return (accuracy_curve, final_global_state).

    `model_factory` builds a fresh model. Errors propagate: a failed run must not
    look like a short but successful one.
    """
    print(f"\n=== EXPERIMENT: {name} ===")
    print(
        f"clients={num_clients}, rounds={rounds}, local_epochs={local_epochs}, lr={local_lr}, "
        f"attack={use_attack}, defense={use_defense}, trim_ratio={trim_ratio}, "
        f"dp={use_dp} (sigma={dp_sigma}, start={dp_start_round}), pos_weight={pos_weight}"
    )

    global_model = model_factory().to(device)
    global_state = deepcopy(global_model.state_dict())
    client_models = [model_factory().to(device) for _ in range(num_clients)]

    acc_curve = []
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

    return acc_curve, deepcopy(global_state)


def ensure_unique_dir(base_dir):
    """Return base_dir if unused, otherwise a timestamped sibling path."""
    base_dir = str(base_dir)
    if not os.path.exists(base_dir):
        return base_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{base_dir}_{stamp}"
    i = 1
    while os.path.exists(candidate):
        i += 1
        candidate = f"{base_dir}_{stamp}_{i}"
    return candidate


def _curve_frame(curve):
    return pd.DataFrame({"round": list(range(1, len(curve) + 1)), "accuracy": curve})


def save_results(out_dir, curves_dict, meta_text, xlsx_name):
    os.makedirs(out_dir, exist_ok=True)
    non_empty = {name: curve for name, curve in curves_dict.items() if curve}

    for name, curve in non_empty.items():
        _curve_frame(curve).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(meta_text)

    if non_empty:
        with pd.ExcelWriter(os.path.join(out_dir, xlsx_name), engine="openpyxl") as writer:
            for name, curve in non_empty.items():
                _curve_frame(curve).to_excel(writer, sheet_name=name[:31], index=False)


def save_plot(out_dir, curves_dict, title, png_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_plot] matplotlib not available, skipping plot generation.")
        return

    non_empty = {name: curve for name, curve in curves_dict.items() if curve}
    if not non_empty:
        return

    plt.figure()
    for name, curve in non_empty.items():
        plt.plot(list(range(1, len(curve) + 1)), curve, marker="o", label=name)
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, png_name), dpi=200, bbox_inches="tight")
    plt.close()
