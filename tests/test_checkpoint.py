import pytest
import torch

from config import REFERENCE_RUN, REFERENCE_SVD_RANK, RESULTS_DIR
from models.hybrid_qnn import HybridQNN
from utils.checkpoints import load_checkpoint, save_checkpoint

REFERENCE_CKPT = RESULTS_DIR / REFERENCE_RUN / f"model_clean_svd_r{REFERENCE_SVD_RANK}.pt"


def test_save_load_roundtrip_preserves_outputs(tmp_path):
    torch.manual_seed(0)
    model = HybridQNN(in_dim=6, n_qubits=3, vqc_layers=2).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model.state_dict(), in_dim=6, threshold=0.4, n_qubits=3, vqc_layers=2)

    loaded, ckpt = load_checkpoint(path)
    assert ckpt["threshold"] == 0.4
    assert not ckpt["svd_enabled"]

    x = torch.randn(4, 6)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_svd_projection_is_stored_in_checkpoint(tmp_path):
    model = HybridQNN(in_dim=3)
    path = tmp_path / "svd.pt"
    save_checkpoint(path, model.state_dict(), in_dim=3, svd_mean=torch.zeros(10),
                    svd_components=torch.eye(10)[:, :3], svd_rank=3)
    _, ckpt = load_checkpoint(path)
    assert ckpt["svd_enabled"] and ckpt["svd_rank"] == 3
    assert ckpt["svd_components"].shape == (10, 3)


def test_invalid_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


@pytest.mark.skipif(not REFERENCE_CKPT.exists(), reason="reference run not present")
def test_reference_checkpoint_loads_and_infers():
    model, ckpt = load_checkpoint(REFERENCE_CKPT)
    assert ckpt["svd_enabled"] and ckpt["in_dim"] == REFERENCE_SVD_RANK

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.randn(5, ckpt["in_dim"])))
    assert probs.shape == (5,)
    assert ((probs >= 0) & (probs <= 1)).all()
