"""Central paths and project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "nsl_kdd"
RESULTS_DIR = ROOT
WEB_DIR = ROOT / "web"

REFERENCE_RUN = "results_svd_hybrid_r16_20260402_030346"
REFERENCE_SVD_RANK = 16

# Label convention used by data/loader.py: "normal" -> 1, every attack -> 0.
# Models are trained with BCEWithLogits on this target, so sigmoid(logit) = P(normal).
ATTACK_LABEL = 0
NORMAL_LABEL = 1
LABELS = {ATTACK_LABEL: "Attack", NORMAL_LABEL: "Normal"}

DEFAULT_N_QUBITS = 4
DEFAULT_VQC_LAYERS = 3
