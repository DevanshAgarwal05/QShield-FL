import torch

from federated.aggregation import fedavg, trimmed_mean
from federated.attacks import sign_flip
from federated.client import apply_update, get_update


def _updates(values):
    return [{"w": torch.tensor([v, 2 * v]), "b": torch.tensor([[v]])} for v in values]


def test_fedavg_is_the_mean():
    agg = fedavg(_updates([1.0, 2.0, 3.0]))
    assert torch.allclose(agg["w"], torch.tensor([2.0, 4.0]))
    assert agg["b"].shape == (1, 1)


def test_trimmed_mean_discards_extremes():
    # 4 clients, trim_ratio=0.25 -> drop 1 min and 1 max per coordinate.
    agg = trimmed_mean(_updates([1.0, 2.0, 3.0, 1000.0]), trim_ratio=0.25)
    assert torch.allclose(agg["w"], torch.tensor([2.5, 5.0]))
    assert agg["b"].shape == (1, 1)


def test_trimmed_mean_resists_sign_flip_attacker_better_than_fedavg():
    honest = _updates([1.0] * 8)
    attacker = sign_flip(honest[0], scale=5.0)
    updates = honest + [attacker]

    err_avg = (fedavg(updates)["w"] - honest[0]["w"]).abs().max()
    err_trim = (trimmed_mean(updates, trim_ratio=0.25)["w"] - honest[0]["w"]).abs().max()
    assert err_trim < err_avg
    assert err_trim < 1e-6


def test_trimmed_mean_falls_back_to_plain_mean_when_nothing_would_remain():
    agg = trimmed_mean(_updates([1.0, 3.0]), trim_ratio=0.5)
    assert torch.allclose(agg["w"], torch.tensor([2.0, 4.0]))


def test_update_roundtrip():
    g = {"w": torch.tensor([1.0, 2.0])}
    local = {"w": torch.tensor([1.5, 1.0])}
    assert torch.allclose(apply_update(g, get_update(g, local))["w"], local["w"])
