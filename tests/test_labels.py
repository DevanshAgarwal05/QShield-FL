"""Backend label handling: 0 = Attack, 1 = Normal, model output = P(Normal)."""
import torch

from backend import app as backend
from config import ATTACK_LABEL, LABELS, NORMAL_LABEL


class _FixedProb(torch.nn.Module):
    """Stub model whose sigmoid output is a fixed P(Normal) per input row (taken from column 0)."""

    def forward(self, x):
        return torch.logit(x[:, 0].clamp(1e-4, 1 - 1e-4))


def test_label_constants_match_loader_convention():
    assert LABELS == {ATTACK_LABEL: "Attack", NORMAL_LABEL: "Normal"}
    assert (ATTACK_LABEL, NORMAL_LABEL) == (0, 1)


def test_pred_label_follows_p_normal():
    assert backend._pred_label(0.9, 0.5) == NORMAL_LABEL
    assert backend._pred_label(0.1, 0.5) == ATTACK_LABEL


def test_pick_indices_uses_loader_labels():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    assert sorted(backend._pick_indices(y, 10, "attack")) == [0, 3]
    assert sorted(backend._pick_indices(y, 10, "normal")) == [1, 2, 4]


def test_confusion_uses_attack_as_positive_class():
    # Rows: (P(Normal), true label)
    X = torch.tensor([[0.05], [0.05], [0.95], [0.95], [0.95]])
    y = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])  # attack, normal, normal, normal, attack
    c = backend._confusion_for_model(_FixedProb(), 0.5, X, y, list(range(5)), model_in_dim=1)
    assert c == {"tp": 1, "fp": 1, "tn": 2, "fn": 1}
    assert backend._acc_from_conf(c) == 0.6
