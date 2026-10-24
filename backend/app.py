from __future__ import annotations

import json
import random
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.hybrid_qnn import HybridQNN
from data.loader import get_binary_tabular_loaders

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
PRIMARY_SVD_RANK = 16
PRIMARY_SVD_FOLDER_PREFIX = f"results_svd_hybrid_r{PRIMARY_SVD_RANK}"
PRIMARY_FINALIZED_FOLDER = "results_svd_hybrid_r16_20260402_030346"

app = FastAPI(title="QSHIELD-FL Dashboard", version="FINAL")

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ----------------------------
# Helpers: safe paths
# ----------------------------
def _is_result_folder(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name.lower()
    return name.startswith("results_") or name.startswith("sanity")


def _safe_resolve_folder(folder_name: str) -> Path:
    candidate = (ROOT / folder_name).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
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
# Model state (in-memory)
# ----------------------------
LOADED: Dict[str, Any] = {
    "folder": None,
    "model_file": None,
    "in_dim": None,
    "raw_in_dim": None,
    "threshold": 0.5,
    "label_names": {"0": "Normal", "1": "Attack"},
    "model_type": "Baseline HQNN",
    "svd_enabled": False,
    "svd_rank": None,
    "svd_mean": None,
    "svd_components": None,
    "meta_file": None,
    "model": None,
}
# ✅ enforce always
LOADED["label_names"] = {"0": "Normal", "1": "Attack"}


def _meta_candidates(folder_path: Path, model_file: Optional[str] = None) -> List[Path]:
    candidates: List[Path] = []

    if model_file:
        m = re.search(r"_svd_r(\d+)", model_file.lower())
        if m:
            candidates.append(folder_path / f"model_meta_svd_r{m.group(1)}.json")

    candidates.append(folder_path / f"model_meta_svd_r{PRIMARY_SVD_RANK}.json")
    candidates.append(folder_path / "model_meta.json")

    svd_meta_files = sorted(folder_path.glob("model_meta_svd_r*.json"), key=lambda p: p.name.lower())
    candidates.extend(svd_meta_files)

    # preserve order while removing duplicates
    uniq: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _load_meta(folder_path: Path, model_file: Optional[str] = None) -> None:
    for meta_path in _meta_candidates(folder_path, model_file=model_file):
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "in_dim" in meta:
                LOADED["in_dim"] = int(meta["in_dim"])
            if "model_in_dim_after_svd" in meta:
                LOADED["in_dim"] = int(meta["model_in_dim_after_svd"])
            if "threshold" in meta:
                LOADED["threshold"] = float(meta["threshold"])
            LOADED["model_type"] = "SVD Hybrid QNN" if bool(meta.get("svd_enabled")) else "Baseline HQNN"
            LOADED["svd_enabled"] = bool(meta.get("svd_enabled", False))
            LOADED["svd_rank"] = int(meta["svd_rank"]) if meta.get("svd_rank") is not None else None
            LOADED["meta_file"] = meta_path.name
            break
        except Exception:
            continue
    # ✅ force names no matter what meta says
    LOADED["label_names"] = {"0": "Normal", "1": "Attack"}


# ----------------------------
# Cache: preprocessed test set for random samples
# ----------------------------
TEST_CACHE: Dict[str, Any] = {
    "ready": False,
    "use_20_percent": None,
    "X": None,  # torch.Tensor [N, in_dim]
    "y": None,  # torch.Tensor [N]
    "in_dim": None,
}


def _ensure_test_cache(use_20_percent: bool):
    if TEST_CACHE["ready"] and TEST_CACHE["use_20_percent"] == use_20_percent:
        return

    try:
        _, test_loader, in_dim = get_binary_tabular_loaders(
            num_clients=3,
            batch_size=256,
            dataset="nsl_kdd",
            data_dir="data/nsl_kdd",
            seed=42,
            split="iid",
            use_20_percent=use_20_percent,
        )
    except Exception:
        alt = not use_20_percent
        _, test_loader, in_dim = get_binary_tabular_loaders(
            num_clients=3,
            batch_size=256,
            dataset="nsl_kdd",
            data_dir="data/nsl_kdd",
            seed=42,
            split="iid",
            use_20_percent=alt,
        )
        use_20_percent = alt

    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    TEST_CACHE["ready"] = True
    TEST_CACHE["use_20_percent"] = use_20_percent
    TEST_CACHE["X"] = X
    TEST_CACHE["y"] = y
    TEST_CACHE["in_dim"] = int(in_dim)


# ----------------------------
# Internal: load a model from checkpoint WITHOUT touching LOADED
# ----------------------------
def _load_model_from_ckpt(
    model_path: Path,
    fallback_in_dim: Optional[int] = None,
    fallback_threshold: float = 0.5
) -> Tuple[HybridQNN, int, float, Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint format; expected dict with 'state_dict'.")

    in_dim = int(ckpt.get("in_dim") or (fallback_in_dim or 0))
    if in_dim <= 0:
        raise ValueError("in_dim missing in checkpoint (and no fallback).")

    threshold = float(ckpt.get("threshold", fallback_threshold))

    model = HybridQNN(in_dim=in_dim, n_qubits=4, vqc_layers=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, in_dim, threshold, ckpt


def _prepare_features(
    feats: List[float],
    model_in_dim: int,
    svd_enabled: bool,
    svd_components: Optional[torch.Tensor],
    svd_mean: Optional[torch.Tensor],
    raw_in_dim: Optional[int],
) -> torch.Tensor:
    n_feats = len(feats)
    x = torch.tensor(feats, dtype=torch.float32)

    if not svd_enabled:
        if n_feats != model_in_dim:
            raise HTTPException(status_code=400, detail=f"Expected {model_in_dim} features, got {n_feats}")
        return x

    if n_feats == model_in_dim:
        return x

    if svd_components is None:
        raise HTTPException(status_code=400, detail="SVD model loaded without projection components.")

    expected_raw = raw_in_dim or int(svd_components.shape[0])
    if n_feats != expected_raw:
        raise HTTPException(
            status_code=400,
            detail=f"SVD model expects either {model_in_dim} projected features or {expected_raw} raw features, got {n_feats}"
        )

    mean_vec = svd_mean if svd_mean is not None else torch.zeros(expected_raw, dtype=torch.float32)
    return torch.matmul((x - mean_vec), svd_components)


def _predict_prob_attack(
    model: HybridQNN,
    feats: List[float],
    model_in_dim: int,
    svd_enabled: bool = False,
    svd_components: Optional[torch.Tensor] = None,
    svd_mean: Optional[torch.Tensor] = None,
    raw_in_dim: Optional[int] = None,
) -> float:
    x_proj = _prepare_features(
        feats=feats,
        model_in_dim=model_in_dim,
        svd_enabled=svd_enabled,
        svd_components=svd_components,
        svd_mean=svd_mean,
        raw_in_dim=raw_in_dim,
    )
    x1 = x_proj.unsqueeze(0)
    with torch.no_grad():
        logits = model(x1)
        return float(torch.sigmoid(logits).item())


def _pick_indices(y: torch.Tensor, n: int, target: str) -> List[int]:
    indices = list(range(len(y)))
    if target == "normal":
        indices = [i for i in indices if int(y[i].item()) == 0]
    elif target == "attack":
        indices = [i for i in indices if int(y[i].item()) == 1]
    elif target != "any":
        raise HTTPException(status_code=400, detail="target must be any|normal|attack")

    if not indices:
        raise HTTPException(status_code=404, detail="No samples found for chosen target.")

    k = min(n, len(indices))
    return random.sample(indices, k=k)


def _confusion_for_model(
    model: HybridQNN,
    threshold: float,
    X: torch.Tensor,
    y: torch.Tensor,
    chosen: List[int],
    model_in_dim: int,
    svd_enabled: bool = False,
    svd_components: Optional[torch.Tensor] = None,
    svd_mean: Optional[torch.Tensor] = None,
    raw_in_dim: Optional[int] = None,
) -> Dict[str, int]:
    tn = fp = fn = tp = 0
    for idx in chosen:
        feats = X[idx].tolist()
        true_label = int(y[idx].item())

        p = _predict_prob_attack(
            model=model,
            feats=feats,
            model_in_dim=model_in_dim,
            svd_enabled=svd_enabled,
            svd_components=svd_components,
            svd_mean=svd_mean,
            raw_in_dim=raw_in_dim,
        )
        pred = 1 if p >= threshold else 0

        if true_label == 0 and pred == 0: tn += 1
        if true_label == 0 and pred == 1: fp += 1
        if true_label == 1 and pred == 0: fn += 1
        if true_label == 1 and pred == 1: tp += 1

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _acc_from_conf(c: Dict[str, int]) -> float:
    total = c["tn"] + c["fp"] + c["fn"] + c["tp"]
    if total <= 0:
        return 0.0
    return (c["tn"] + c["tp"]) / total


def _preferred_model_name(model_names: List[str]) -> Optional[str]:
    if not model_names:
        return None

    lowered = {n.lower(): n for n in model_names}
    preferred = [
        f"model_clean_svd_r{PRIMARY_SVD_RANK}.pt",
        f"model_defended_svd_r{PRIMARY_SVD_RANK}.pt",
        f"model_defended_dp_svd_r{PRIMARY_SVD_RANK}.pt",
        f"model_attack_svd_r{PRIMARY_SVD_RANK}.pt",
        "model_clean.pt",
        "model_defended.pt",
        "model_defended_dp.pt",
        "model_attack.pt",
    ]
    for p in preferred:
        if p in lowered:
            return lowered[p]
    return sorted(model_names)[0]


def _discover_default_results_folder() -> Dict[str, Any]:
    folders: List[Dict[str, Any]] = []
    for p in ROOT.iterdir():
        if _is_result_folder(p):
            folders.append({"name": p.name, "mtime": p.stat().st_mtime})

    folders.sort(key=lambda x: x["mtime"], reverse=True)
    if not folders:
        return {"folder": None, "ambiguous": False, "candidates": []}

    finalized = next((x for x in folders if x["name"] == PRIMARY_FINALIZED_FOLDER), None)
    if finalized:
        return {
            "folder": finalized["name"],
            "ambiguous": False,
            "candidates": [finalized["name"]],
            "source": "explicit_finalized",
        }

    svd_r16 = [
        x for x in folders
        if x["name"].lower().startswith(PRIMARY_SVD_FOLDER_PREFIX.lower())
    ]
    if svd_r16:
        return {
            "folder": svd_r16[0]["name"],
            "ambiguous": len(svd_r16) > 1,
            "candidates": [x["name"] for x in svd_r16],
            "source": "latest_svd_r16",
        }

    return {
        "folder": folders[0]["name"],
        "ambiguous": False,
        "candidates": [folders[0]["name"]],
        "source": "latest_any_results",
    }


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
    folders = []
    for p in ROOT.iterdir():
        if _is_result_folder(p):
            mtime = p.stat().st_mtime if p.exists() else 0.0
            folders.append({"name": p.name, "mtime": mtime})
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
    d = _discover_default_results_folder()
    folder_name = d.get("folder")
    if not folder_name:
        return {
            "folder": None,
            "model_file": None,
            "meta_file": None,
            "model_type": "SVD Hybrid QNN",
            "svd_rank": PRIMARY_SVD_RANK,
            "ambiguous": False,
            "candidates": [],
        }

    folder_path = _safe_resolve_folder(folder_name)
    model_files = sorted(
        [p.name for p in folder_path.iterdir() if p.is_file() and p.name.lower().startswith("model_") and p.name.lower().endswith(".pt")]
    )
    model_file = _preferred_model_name(model_files)

    meta_file = None
    for mp in _meta_candidates(folder_path, model_file=model_file):
        if mp.exists():
            meta_file = mp.name
            break

    return {
        "folder": folder_name,
        "model_file": model_file,
        "meta_file": meta_file,
        "model_type": "SVD Hybrid QNN",
        "svd_rank": PRIMARY_SVD_RANK,
        "ambiguous": bool(d.get("ambiguous")),
        "candidates": d.get("candidates", []),
        "source": d.get("source"),
        "preferred_finalized_folder": PRIMARY_FINALIZED_FOLDER,
    }


@app.get("/api/results/download")
def download_file(folder: str = Query(...), file: str = Query(...)):
    folder_path = _safe_resolve_folder(folder)
    file_path = _safe_resolve_file(folder_path, file)
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@app.get("/api/results/curves")
def curves(folder: str = Query(...)):
    folder_path = _safe_resolve_folder(folder)

    def read_curve(csv_path: Path):
        rounds, accs = [], []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rounds.append(int(float(row["round"])))
                accs.append(float(row["accuracy"]))
        return rounds, accs

    out = {}
    for name in ["clean", "attack", "defended", "defended_dp"]:
        p = folder_path / f"{name}.csv"
        if p.exists():
            r, a = read_curve(p)
            out[name] = {
                "rounds": r,
                "acc": a,
                "final": round(a[-1], 4) if a else None
            }

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
        "label_names": LOADED["label_names"],
        "model_type": LOADED["model_type"],
        "svd_enabled": LOADED["svd_enabled"],
        "svd_rank": LOADED["svd_rank"],
        "meta_file": LOADED["meta_file"],
    }


@app.post("/api/model/load")
def load_model(folder: str = Query(...), model_file: str = Query(...)):
    folder_path = _safe_resolve_folder(folder)
    model_path = _safe_resolve_file(folder_path, model_file)

    _load_meta(folder_path, model_file=model_path.name)

    # use meta as fallback if ckpt missing
    model, in_dim, threshold, ckpt = _load_model_from_ckpt(
        model_path,
        fallback_in_dim=LOADED.get("in_dim"),
        fallback_threshold=float(LOADED.get("threshold", 0.5)),
    )

    svd_enabled = bool(ckpt.get("svd_enabled", False))
    svd_rank = ckpt.get("svd_rank")
    svd_mean = ckpt.get("svd_mean")
    svd_components = ckpt.get("svd_components")
    raw_in_dim = None
    if isinstance(svd_components, torch.Tensor):
        svd_components = svd_components.float()
        raw_in_dim = int(svd_components.shape[0])
    if isinstance(svd_mean, torch.Tensor):
        svd_mean = svd_mean.float()

    LOADED["folder"] = folder_path.name
    LOADED["model_file"] = model_path.name
    LOADED["in_dim"] = int(in_dim)
    LOADED["raw_in_dim"] = int(raw_in_dim) if raw_in_dim is not None else int(in_dim)
    LOADED["threshold"] = float(threshold)
    LOADED["model"] = model
    LOADED["label_names"] = {"0": "Normal", "1": "Attack"}
    LOADED["svd_enabled"] = bool(svd_enabled)
    LOADED["svd_rank"] = int(svd_rank) if svd_rank is not None else LOADED.get("svd_rank")
    LOADED["svd_mean"] = svd_mean
    LOADED["svd_components"] = svd_components
    LOADED["model_type"] = "SVD Hybrid QNN" if LOADED["svd_enabled"] else "Baseline HQNN"

    return model_status()


@app.post("/api/predict")
def predict(payload: Dict[str, Any]):
    if LOADED["model"] is None or LOADED["in_dim"] is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load model_*.pt first.")

    feats = payload.get("features")
    if not isinstance(feats, list):
        raise HTTPException(status_code=400, detail="Payload must be: { features: [..] }")

    prob_attack = _predict_prob_attack(
        model=LOADED["model"],
        feats=feats,
        model_in_dim=int(LOADED["in_dim"]),
        svd_enabled=bool(LOADED["svd_enabled"]),
        svd_components=LOADED.get("svd_components"),
        svd_mean=LOADED.get("svd_mean"),
        raw_in_dim=LOADED.get("raw_in_dim"),
    )

    pred = 1 if prob_attack >= LOADED["threshold"] else 0
    label = LOADED["label_names"].get(str(pred), f"Class {pred}")

    # ✅ return BOTH keys so older/newer frontend never breaks
    return JSONResponse({
        "prob_attack": float(prob_attack),
        "probability_class1": float(prob_attack),
        "threshold": float(LOADED["threshold"]),
        "pred_class": int(pred),
        "pred_label": label,
        "model_type": LOADED["model_type"],
        "svd_enabled": LOADED["svd_enabled"],
        "svd_rank": LOADED["svd_rank"],
    })


# ----------------------------
# Random sample from test set (preprocessed)
# ----------------------------
@app.get("/api/test/random")
def random_test_sample(
    target: str = Query("any", description="any|normal|attack"),
    use_20_percent: int = Query(0, description="0=full, 1=20pct"),
):
    use_20 = bool(use_20_percent)
    _ensure_test_cache(use_20)

    X = TEST_CACHE["X"]
    y = TEST_CACHE["y"]
    in_dim = TEST_CACHE["in_dim"]

    if X is None or y is None:
        raise HTTPException(status_code=500, detail="Test cache not loaded.")

    indices = _pick_indices(y, n=1, target=target)
    idx = indices[0]

    feats = X[idx].tolist()
    true_label = int(y[idx].item())
    true_name = LOADED["label_names"].get(str(true_label), f"Class {true_label}")

    return {
        "in_dim": int(in_dim),
        "features": feats,
        "true_label": true_label,
        "true_label_name": true_name,
        "use_20_percent": use_20,
    }


# ----------------------------
# Batch evaluation (n random samples)
# ----------------------------
@app.get("/api/eval/batch")
def batch_eval(
    n: int = Query(50, ge=1, le=500),
    target: str = Query("any", description="any|normal|attack"),
    use_20_percent: int = Query(0, description="0=full, 1=20pct"),
):
    if LOADED["model"] is None or LOADED["in_dim"] is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load model_*.pt first.")

    use_20 = bool(use_20_percent)
    _ensure_test_cache(use_20)

    X = TEST_CACHE["X"]
    y = TEST_CACHE["y"]
    if X is None or y is None:
        raise HTTPException(status_code=500, detail="Test cache not loaded.")

    chosen = _pick_indices(y, n=n, target=target)
    c = _confusion_for_model(
        model=LOADED["model"],
        threshold=float(LOADED["threshold"]),
        X=X,
        y=y,
        chosen=chosen,
        model_in_dim=int(LOADED["in_dim"]),
        svd_enabled=bool(LOADED["svd_enabled"]),
        svd_components=LOADED.get("svd_components"),
        svd_mean=LOADED.get("svd_mean"),
        raw_in_dim=LOADED.get("raw_in_dim"),
    )
    acc = _acc_from_conf(c)

    return {
        "n": int(len(chosen)),
        "target": target,
        "accuracy": round(float(acc), 4),
        "confusion": c,
        "threshold": float(LOADED["threshold"]),
        "labels": LOADED["label_names"],
    }


# ----------------------------
# NEW: Compare models in a result folder (same random sample set)
# ----------------------------
@app.get("/api/eval/compare")
def compare_models(
    folder: str = Query(...),
    n: int = Query(200, ge=10, le=2000),
    target: str = Query("any", description="any|normal|attack"),
    use_20_percent: int = Query(0, description="0=full, 1=20pct"),
):
    folder_path = _safe_resolve_folder(folder)

    use_20 = bool(use_20_percent)
    _ensure_test_cache(use_20)
    X = TEST_CACHE["X"]
    y = TEST_CACHE["y"]
    if X is None or y is None:
        raise HTTPException(status_code=500, detail="Test cache not loaded.")

    chosen = _pick_indices(y, n=n, target=target)

    # discover model files
    model_files = sorted([p.name for p in folder_path.iterdir() if p.is_file() and p.name.lower().startswith("model_") and p.name.lower().endswith(".pt")])
    if not model_files:
        raise HTTPException(status_code=404, detail="No model_*.pt files found in that folder.")

    # fallback from meta if needed
    fallback_in_dim = None
    fallback_threshold = 0.5
    meta_path = folder_path / "model_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            fallback_in_dim = int(meta.get("in_dim")) if meta.get("in_dim") is not None else None
            fallback_threshold = float(meta.get("threshold", 0.5))
        except Exception:
            pass

    rows = []
    for mf in model_files:
        try:
            model_path = _safe_resolve_file(folder_path, mf)
            model, in_dim, threshold, ckpt = _load_model_from_ckpt(
                model_path,
                fallback_in_dim=fallback_in_dim,
                fallback_threshold=fallback_threshold
            )
            svd_enabled = bool(ckpt.get("svd_enabled", False))
            svd_mean = ckpt.get("svd_mean")
            svd_components = ckpt.get("svd_components")
            raw_in_dim = None
            if isinstance(svd_components, torch.Tensor):
                svd_components = svd_components.float()
                raw_in_dim = int(svd_components.shape[0])
            if isinstance(svd_mean, torch.Tensor):
                svd_mean = svd_mean.float()

            c = _confusion_for_model(
                model=model,
                threshold=float(threshold),
                X=X,
                y=y,
                chosen=chosen,
                model_in_dim=int(in_dim),
                svd_enabled=svd_enabled,
                svd_components=svd_components,
                svd_mean=svd_mean,
                raw_in_dim=raw_in_dim,
            )
            acc = _acc_from_conf(c)
            rows.append({
                "model_file": mf,
                "status": "ok",
                "threshold": float(threshold),
                "model_type": "SVD Hybrid QNN" if svd_enabled else "Baseline HQNN",
                "svd_rank": int(ckpt.get("svd_rank")) if ckpt.get("svd_rank") is not None else None,
                "accuracy": round(float(acc), 4),
                "tn": c["tn"], "fp": c["fp"], "fn": c["fn"], "tp": c["tp"],
            })
        except Exception as e:
            rows.append({"model_file": mf, "status": f"error: {str(e)}"})

    # order rows by accuracy desc (ok only)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    bad_rows = [r for r in rows if r.get("status") != "ok"]
    ok_rows.sort(key=lambda r: r.get("accuracy", 0.0), reverse=True)

    return {
        "folder": folder_path.name,
        "n": int(len(chosen)),
        "target": target,
        "labels": LOADED["label_names"],
        "rows": ok_rows + bad_rows
    }
