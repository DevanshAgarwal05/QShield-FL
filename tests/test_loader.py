import numpy as np
import pytest

from config import DATA_DIR
from data.loader import get_binary_tabular_loaders


def _total(loaders):
    return sum(len(dl.dataset) for dl in loaders)


def test_iid_split_covers_all_train_samples():
    loaders, test_loader, in_dim = get_binary_tabular_loaders(
        num_clients=3, dataset="breast_cancer", batch_size=32, split="iid"
    )
    assert len(loaders) == 3
    assert in_dim == 30
    assert _total(loaders) == 455  # 80% of 569
    assert len(test_loader.dataset) == 114


def test_dirichlet_split_keeps_every_sample():
    loaders, _, _ = get_binary_tabular_loaders(
        num_clients=4, dataset="breast_cancer", batch_size=32, split="dirichlet", dirichlet_alpha=0.5
    )
    assert _total(loaders) == 455


def test_labels_are_binary_and_features_float32():
    loaders, test_loader, _ = get_binary_tabular_loaders(num_clients=2, dataset="breast_cancer")
    x, y = loaders[0].dataset.tensors
    assert str(x.dtype) == "torch.float32"
    assert set(np.unique(y.numpy())) <= {0.0, 1.0}


def test_unknown_split_and_dataset_raise():
    with pytest.raises(ValueError):
        get_binary_tabular_loaders(dataset="breast_cancer", split="nope")
    with pytest.raises(ValueError):
        get_binary_tabular_loaders(dataset="nope")


@pytest.mark.skipif(not (DATA_DIR / "KDDTrain+.txt").exists(), reason="NSL-KDD not downloaded")
def test_nsl_kdd_normal_is_label_one():
    """Pins the convention the dashboard depends on: normal -> 1, attack -> 0."""
    _, test_loader, in_dim = get_binary_tabular_loaders(
        num_clients=3, dataset="nsl_kdd", use_20_percent=True
    )
    y = test_loader.dataset.tensors[1].numpy()
    assert in_dim == 122
    # KDDTest+ is ~43% normal and ~57% attack; a swapped mapping would invert this.
    assert 0.35 < y.mean() < 0.5
