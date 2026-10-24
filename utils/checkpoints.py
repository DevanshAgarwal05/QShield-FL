"""Model checkpoint save/load. Checkpoints hold only tensors and plain numbers."""
from datetime import datetime

import torch

from config import DEFAULT_N_QUBITS, DEFAULT_VQC_LAYERS
from models.hybrid_qnn import HybridQNN


def save_checkpoint(path, state_dict, in_dim, threshold=0.5, n_qubits=DEFAULT_N_QUBITS,
                    vqc_layers=DEFAULT_VQC_LAYERS, svd_mean=None, svd_components=None, svd_rank=None):
    ckpt = {
        "state_dict": state_dict,
        "in_dim": int(in_dim),
        "threshold": float(threshold),
        "n_qubits": int(n_qubits),
        "vqc_layers": int(vqc_layers),
        "svd_enabled": svd_components is not None,
        "saved_at": datetime.now().isoformat(),
    }
    if svd_components is not None:
        ckpt["svd_rank"] = int(svd_rank)
        ckpt["svd_mean"] = svd_mean.cpu()
        ckpt["svd_components"] = svd_components.cpu()
    torch.save(ckpt, path)


def load_checkpoint(path, fallback_in_dim=None, fallback_threshold=0.5):
    """Return (model in eval mode, checkpoint dict). Older checkpoints lack n_qubits/vqc_layers
    and fall back to the project defaults."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint format; expected dict with 'state_dict'.")

    in_dim = int(ckpt.get("in_dim") or (fallback_in_dim or 0))
    if in_dim <= 0:
        raise ValueError("in_dim missing in checkpoint (and no fallback).")
    ckpt["in_dim"] = in_dim
    ckpt["threshold"] = float(ckpt.get("threshold", fallback_threshold))

    model = HybridQNN(
        in_dim=in_dim,
        n_qubits=int(ckpt.get("n_qubits", DEFAULT_N_QUBITS)),
        vqc_layers=int(ckpt.get("vqc_layers", DEFAULT_VQC_LAYERS)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
