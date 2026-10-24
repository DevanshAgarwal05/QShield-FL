from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    ATTACK_LABEL,
    DATA_DIR,
    LABELS,
    NORMAL_LABEL,
    REFERENCE_RUN,
    REFERENCE_SVD_RANK,
    RESULTS_DIR,
    ROOT,
    WEB_DIR,
)
from data.loader import get_binary_tabular_loaders
from models.hybrid_qnn import HybridQNN
from utils.checkpoints import load_checkpoint

# Label convention (see config.py): 0 = Attack, 1 = Normal, and the model outputs
# sigmoid(logit) = P(Normal). Everything the API reports as "attack" is derived from that.

app = FastAPI(title="QSHIELD-FL Dashboard", version="FINAL")

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ----------------------------
# Helpers: safe paths
# ----------------------------
def _is_result_folder(p: Path) -> bool:
    return p.is_dir() and p.name.lower().startswith("results_")


def _safe_resolve_folder(folder_name: str) -> Path:
    candidate = (RESULTS_DIR / folder_name).resolve()
    if RESULTS_DIR not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid folder path.")
    if not candidate.exists() or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found.")
    if not _is_result_folder(candidate):
        raise HTTPException(status_code=400, detail="Not a result folder.")
    return candidate


def _safe_resolve_file(folder_path: Path, filename: str) -> Path:
    file_path = (folder_path / filename).resolve()
    if folder_path not in file_path.parents:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return file_path


# ----------------------------
# Model state (in-memory, single worker)
# ----------------------------
LOADED: Dict[str, Any] = {
    "folder": None,
    "model_file": None,
    "in_dim": None,
    "raw_in_dim": None,
    "threshold": 0.5,
    "model_type": "Baseline HQNN",
    "svd_enabled": False,
    "svd_rank": None,
    "svd_mean": None,
    "svd_components": None,
    "model": None,
}


def _label_names() -> Dict[str, str]:
    return {str(k): v for k, v in LABELS.items()}


def _model_type(svd_enabled: bool) -> str:
    return "SVD Hybrid QNN" if svd_enabled else "Baseline HQNN"


# ----------------------------
# Cache: preprocessed test set for random samples
# ----------------------------
TEST_CACHE: Dict[str, Any] = {"use_20_percent": None, "X": None, "y": None}


def _ensure_test_cache(use_20_percent: bool) -> None:
    if TEST_CACHE["X"] is not None and TEST_CACHE["use_20_percent"] == use_20_percent:
        return

    _, test_loader, _ = get_binary_tabular_loaders(
        num_clients=3,
        batch_size=256,
        dataset="nsl_kdd",
        data_dir=str(DATA_DIR),
        seed=42,
        split="iid",
        use_20_percent=use_20_percent,
    )
    TEST_CACHE["X"] = test_loader.dataset.tensors[0].detach().cpu()
    TEST_CACHE["y"] = test_loader.dataset.tensors[1].detach().cpu()
    TEST_CACHE["use_20_percent"] = use_20_percent


def _checkpoint_svd(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    svd_components = ckpt.get("svd_components")
    svd_mean = ckpt.get("svd_mean")
    return {
        "svd_enabled": bool(ckpt.get("svd_enabled", False)),
        "svd_rank": int(ckpt["svd_rank"]) if ckpt.get("svd_rank") is not None else None,
        "svd_components": svd_components.float() if isinstance(svd_components, torch.Tensor) else None,
        "svd_mean": svd_mean.float() if isinstance(svd_mean, torch.Tensor) else None,
    }


def _prepare_features(
    feats: List[float],
    model_in_dim: int,
    svd_components: Optional[torch.Tensor],
    svd_mean: Optional[torch.Tensor],
) -> torch.Tensor:
    """Accept either already-projected features (model_in_dim values) or raw features
    (svd_components.shape[0] values, projected here)."""
    x = torch.tensor(feats, dtype=torch.float32)
    n_feats = x.numel()

    if n_feats == model_in_dim:
        return x

    if svd_components is None:
        raise HTTPException(status_code=400, detail=f"Expected {model_in_dim} features, got {n_feats}")

    expected_raw = int(svd_components.shape[0])
    if n_feats != expected_raw:
        raise HTTPException(
            status_code=400,
            detail=f"SVD model expects either {model_in_dim} projected features or {expected_raw} raw features, got {n_feats}",
        )

    mean_vec = svd_mean if svd_mean is not None else torch.zeros(expected_raw, dtype=torch.float32)
    return torch.matmul(x - mean_vec, svd_components)


def _predict_prob_normal(
    model: HybridQNN,
    feats: List[float],
    model_in_dim: int,
    svd_components: Optional[torch.Tensor] = None,
    svd_mean: Optional[torch.Tensor] = None,
) -> float:
    x = _prepare_features(feats, model_in_dim, svd_components, svd_mean)
    with torch.no_grad():
        return float(torch.sigmoid(model(x.unsqueeze(0))).item())


def _pred_label(prob_normal: float, threshold: float) -> int:
    return NORMAL_LABEL if prob_normal >= threshold else ATTACK_LABEL


def _pick_indices(y: torch.Tensor, n: int, target: str) -> List[int]:
    if target == "normal":
        wanted = NORMAL_LABEL
    elif target == "attack":
        wanted = ATTACK_LABEL
    elif target == "any":
        wanted = None
    else:
        raise HTTPException(status_code=400, detail="target must be any|normal|attack")

    indices = [i for i in range(len(y)) if wanted is None or int(y[i].item()) == wanted]
    if not indices:
        raise HTTPException(status_code=404, detail="No samples found for chosen target.")

    return random.sample(indices, k=min(n, len(indices)))


def _confusion_for_model(
    model: HybridQNN,
    threshold: float,
    X: torch.Tensor,
    y: torch.Tensor,
    chosen: List[int],
    model_in_dim: int,
    svd_components: Optional[torch.Tensor] = None,
    svd_mean: Optional[torch.Tensor] = None,
) -> Dict[str, int]:
    """Confusion counts with Attack as the positive class (tp = attacks caught)."""
    tn = fp = fn = tp = 0
    for idx in chosen:
        true_attack = int(y[idx].item()) == ATTACK_LABEL
        p_normal = _predict_prob_normal(model, X[idx].tolist(), model_in_dim, svd_components, svd_mean)
        pred_attack = _pred_label(p_normal, threshold) == ATTACK_LABEL

        if true_attack and pred_attack:
            tp += 1
        elif true_attack:
            fn += 1
        elif pred_attack:
            fp += 1
        else:
            tn += 1

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _acc_from_conf(c: Dict[str, int]) -> float:
    total = c["tn"] + c["fp"] + c["fn"] + c["tp"]
    return (c["tn"] + c["tp"]) / total if total > 0 else 0.0


def _model_files(folder_path: Path) -> List[str]:
    return sorted(
        p.name for p in folder_path.iterdir()
        if p.is_file() and p.name.lower().startswith("model_") and p.name.lower().endswith(".pt")
    )


def _preferred_model_name(model_names: List[str]) -> Optional[str]:
    if not model_names:
        return None

    lowered = {n.lower(): n for n in model_names}
    preferred = [
        f"model_clean_svd_r{REFERENCE_SVD_RANK}.pt",
        f"model_defended_svd_r{REFERENCE_SVD_RANK}.pt",
        f"model_defended_dp_svd_r{REFERENCE_SVD_RANK}.pt",
        f"model_attack_svd_r{REFERENCE_SVD_RANK}.pt",
        "model_clean.pt",
        "model_defended.pt",
        "model_defended_dp.pt",
        "model_attack.pt",
    ]
    for p in preferred:
        if p in lowered:
            return lowered[p]
    return model_names[0]


def _discover_default_results_folder() -> Optional[str]:
    """Reference run if present, otherwise the newest results_* folder."""
    folders = sorted(
        (p for p in RESULTS_DIR.iterdir() if _is_result_folder(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if any(p.name == REFERENCE_RUN for p in folders):
        return REFERENCE_RUN
    return folders[0].name if folders else None


# ----------------------------
# Pages
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    index = WEB_DIR / "index.html"
    if not index.exists():
        return "<h2>web/index.html not found</h2>"
    return index.read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> str:
    admin = WEB_DIR / "admin.html"
    if not admin.exists():
        return "<h2>web/admin.html not found</h2>"
    return admin.read_text(encoding="utf-8")


# ----------------------------
# API: results browsing
# ----------------------------
@app.get("/api/results/folders")
def list_result_folders() -> Dict[str, Any]:
    folders = [{"name": p.name, "mtime": p.stat().st_mtime} for p in RESULTS_DIR.iterdir() if _is_result_folder(p)]
    folders.sort(key=lambda x: x["mtime"], reverse=True)
    return {"root": str(ROOT), "folders": folders}


@app.get("/api/results/files")
def list_files(folder: str = Query(...)) -> Dict[str, Any]:
    folder_path = _safe_resolve_folder(folder)
    files: List[Dict[str, Any]] = []
    for f in folder_path.iterdir():
        if f.is_file():
            st = f.stat()
            files.append({
                "name": f.name,
                "size_bytes": st.st_size,
                "mtime": st.st_mtime,
                "ext": f.suffix.lower().lstrip("."),
            })
    files.sort(key=lambda x: x["name"].lower())
    return {"folder": folder_path.name, "files": files}


@app.get("/api/results/default")
def default_results() -> Dict[str, Any]:
    folder_name = _discover_default_results_folder()
    if not folder_name:
        return {
            "folder": None,
            "model_file": None,
            "meta_file": None,
            "model_type": _model_type(True),
            "svd_rank": REFERENCE_SVD_RANK,
        }

    folder_path = _safe_resolve_folder(folder_name)
    model_file = _preferred_model_name(_model_files(folder_path))

    meta_file = None
    for name in (f"model_meta_svd_r{REFERENCE_SVD_RANK}.json", "model_meta.json"):
        if (folder_path / name).exists():
            meta_file = name
            break

    return {
        "folder": folder_name,
        "model_file": model_file,
        "meta_file": meta_file,
        "model_type": _model_type(True),
        "svd_rank": REFERENCE_SVD_RANK,
        "preferred_finalized_folder": REFERENCE_RUN,
    }


@app.get("/api/results/download")
def download_file(folder: str = Query(...), file: str = Query(...)):
    file_path = _safe_resolve_file(_safe_resolve_folder(folder), file)
    return FileResponse(path=str(file_path), filename=file_path.name, media_type="application/octet-stream")


@app.get("/api/results/curves")
def curves(folder: str = Query(...)):
    folder_path = _safe_resolve_folder(folder)

    def read_curve(csv_path: Path):
        rounds, accs = [], []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rounds.append(int(float(row["round"])))
                accs.append(float(row["accuracy"]))
        return rounds, accs

    out = {}
    for name in ["clean", "attack", "defended", "defended_dp"]:
        p = folder_path / f"{name}.csv"
        if p.exists():
            r, a = read_curve(p)
            out[name] = {"rounds": r, "acc": a, "final": round(a[-1], 4) if a else None}

    return {"folder": folder_path.name, "curves": out}


# ----------------------------
# API: model load / status / predict
# ----------------------------
@app.get("/api/model/status")
def model_status():
    return {
        "loaded": LOADED["model"] is not None,
        "folder": LOADED["folder"],
        "model_file": LOADED["model_file"],
        "in_dim": LOADED["in_dim"],
        "raw_in_dim": LOADED["raw_in_dim"],
        "threshold": LOADED["threshold"],
        "label_names": _label_names(),
        "model_type": LOADED["model_type"],
        "svd_enabled": LOADED["svd_enabled"],
        "svd_rank": LOADED["svd_rank"],
    }


def _fallback_from_meta(folder_path: Path) -> Dict[str, Any]:
    """in_dim/threshold from model_meta.json for old checkpoints that lack them."""
    meta_path = folder_path / "model_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@app.post("/api/model/load")
def load_model(folder: str = Query(...), model_file: str = Query(...)):
    folder_path = _safe_resolve_folder(folder)
    model_path = _safe_resolve_file(folder_path, model_file)

    meta = _fallback_from_meta(folder_path)
    try:
        model, ckpt = load_checkpoint(
            model_path,
            fallback_in_dim=meta.get("in_dim"),
            fallback_threshold=float(meta.get("threshold", 0.5)),
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Could not load checkpoint: {e}")

    svd = _checkpoint_svd(ckpt)
    raw_in_dim = int(svd["svd_components"].shape[0]) if svd["svd_components"] is not None else int(ckpt["in_dim"])

    LOADED.update(
        folder=folder_path.name,
        model_file=model_path.name,
        in_dim=int(ckpt["in_dim"]),
        raw_in_dim=raw_in_dim,
        threshold=float(ckpt["threshold"]),
        model=model,
        model_type=_model_type(svd["svd_enabled"]),
        svd_enabled=svd["svd_enabled"],
        svd_rank=svd["svd_rank"],
        svd_mean=svd["svd_mean"],
        svd_components=svd["svd_components"],
    )
    return model_status()


@app.post("/api/predict")
def predict(payload: Dict[str, Any]):
    if LOADED["model"] is None or LOADED["in_dim"] is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load model_*.pt first.")

    feats = payload.get("features")
    if not isinstance(feats, list):
        raise HTTPException(status_code=400, detail="Payload must be: { features: [..] }")

    prob_normal = _predict_prob_normal(
        model=LOADED["model"],
        feats=feats,
        model_in_dim=int(LOADED["in_dim"]),
        svd_components=LOADED["svd_components"],
        svd_mean=LOADED["svd_mean"],
    )
    pred = _pred_label(prob_normal, float(LOADED["threshold"]))

    return JSONResponse({
        "prob_attack": 1.0 - prob_normal,
        "prob_normal": prob_normal,
        "probability_class1": prob_normal,  # P(label 1) = P(Normal); kept for older clients
        "threshold": float(LOADED["threshold"]),  # applied to P(Normal)
        "pred_class": int(pred),
        "pred_label": LABELS[pred],
        "model_type": LOADED["model_type"],
        "svd_enabled": LOADED["svd_enabled"],
        "svd_rank": LOADED["svd_rank"],
    })


# ----------------------------
# Random sample / batch evaluation on the preprocessed test set
# ----------------------------
@app.get("/api/test/random")
def random_test_sample(
    target: str = Query("any", description="any|normal|attack"),
    use_20_percent: int = Query(0, description="0=full, 1=20pct"),
):
    use_20 = bool(use_20_percent)
    _ensure_test_cache(use_20)

    X, y = TEST_CACHE["X"], TEST_CACHE["y"]
    idx = _pick_indices(y, n=1, target=target)[0]
    true_label = int(y[idx].item())

    return {
        "in_dim": int(X.shape[1]),
        "features": X[idx].tolist(),
        "true_label": true_label,
        "true_label_name": LABELS.get(true_label, f"Class {true_label}"),
        "use_20_percent": use_20,
    }


@app.get("/api/eval/batch")
def batch_eval(
    n: int = Query(50, ge=1, le=500),
    target: str = Query("any", description="any|normal|attack"),
    use_20_percent: int = Query(0, description="0=full, 1=20pct"),
):
    if LOADED["model"] is None or LOADED["in_dim"] is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load model_*.pt first.")

    _ensure_test_cache(bool(use_20_percent))
    X, y = TEST_CACHE["X"], TEST_CACHE["y"]

    chosen = _pick_indices(y, n=n, target=target)
    c = _confusion_for_model(
        model=LOADED["model"],
        threshold=float(LOADED["threshold"]),
        X=X,
        y=y,
        chosen=chosen,
        model_in_dim=int(LOADED["in_dim"]),
        svd_components=LOADED["svd_components"],
        svd_mean=LOADED["svd_mean"],
    )

    return {
        "n": len(chosen),
        "target": target,
        "accuracy": round(_acc_from_conf(c), 4),
        "confusion": c,
        "threshold": float(LOADED["threshold"]),
        "labels": _label_names(),
    }


@app.get("/api/eval/compare")
def compare_models(
    folder: str = Query(...),
    n: int = Query(200, ge=10, le=2000),
    target: str = Query("any", description="any|normal|attack"),
    use_20_percent: int = Query(0, description="0=full, 1=20pct"),
):
    """Evaluate every model_*.pt in a result folder on the same random sample set."""
    folder_path = _safe_resolve_folder(folder)

    _ensure_test_cache(bool(use_20_percent))
    X, y = TEST_CACHE["X"], TEST_CACHE["y"]
    chosen = _pick_indices(y, n=n, target=target)

    model_files = _model_files(folder_path)
    if not model_files:
        raise HTTPException(status_code=404, detail="No model_*.pt files found in that folder.")

    meta = _fallback_from_meta(folder_path)

    rows = []
    for mf in model_files:
        try:
            model, ckpt = load_checkpoint(
                _safe_resolve_file(folder_path, mf),
                fallback_in_dim=meta.get("in_dim"),
                fallback_threshold=float(meta.get("threshold", 0.5)),
            )
            svd = _checkpoint_svd(ckpt)
            c = _confusion_for_model(
                model=model,
                threshold=ckpt["threshold"],
                X=X,
                y=y,
                chosen=chosen,
                model_in_dim=ckpt["in_dim"],
                svd_components=svd["svd_components"],
                svd_mean=svd["svd_mean"],
            )
            rows.append({
                "model_file": mf,
                "status": "ok",
                "threshold": ckpt["threshold"],
                "model_type": _model_type(svd["svd_enabled"]),
                "svd_rank": svd["svd_rank"],
                "accuracy": round(_acc_from_conf(c), 4),
                **c,
            })
        except Exception as e:
            rows.append({"model_file": mf, "status": f"error: {e}"})

    ok_rows = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: r["accuracy"], reverse=True)
    bad_rows = [r for r in rows if r["status"] != "ok"]

    return {
        "folder": folder_path.name,
        "n": len(chosen),
        "target": target,
        "labels": _label_names(),
        "rows": ok_rows + bad_rows,
    }
