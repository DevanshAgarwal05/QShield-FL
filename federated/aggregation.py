import torch

def fedavg(updates):
    # updates: list of state_dict-like tensors (same keys)
    agg = {}
    keys = updates[0].keys()
    for k in keys:
        agg[k] = torch.stack([u[k] for u in updates], dim=0).mean(dim=0)
    return agg

def trimmed_mean(updates, trim_ratio=0.2):
    """
    Coordinate-wise trimmed mean.
    If 5 clients, trim_ratio=0.2 trims 1 min + 1 max -> robust.
    """
    agg = {}
    keys = updates[0].keys()
    n = len(updates)
    trim_k = int(n * trim_ratio)

    for k in keys:
        stacked = torch.stack([u[k] for u in updates], dim=0)  # (n, ...)
        flat = stacked.view(n, -1)
        sorted_vals, _ = torch.sort(flat, dim=0)
        trimmed = sorted_vals[trim_k:n-trim_k, :] if (n - 2*trim_k) > 0 else sorted_vals
        mean = trimmed.mean(dim=0)
        agg[k] = mean.view_as(updates[0][k])

    return agg
