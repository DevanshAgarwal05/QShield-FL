import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


def _split_clients_iid(X, y, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    chunks = np.array_split(idx, num_clients)
    return [(X[c], y[c]) for c in chunks]


def _split_clients_dirichlet(X, y, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        counts = (proportions * len(idx_c)).astype(int)

        diff = len(idx_c) - counts.sum()
        if diff != 0:
            counts[np.argmax(counts)] += diff

        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(idx_c[start:end].tolist())
            start = end

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return [(X[np.array(ci)], y[np.array(ci)]) for ci in client_indices]


def _make_loaders_from_arrays(X_train, y_train, X_test, y_test, num_clients, batch_size, seed=42, split="iid", dirichlet_alpha=0.5):
    if split == "iid":
        client_splits = _split_clients_iid(X_train, y_train, num_clients=num_clients, seed=seed)
    elif split == "dirichlet":
        client_splits = _split_clients_dirichlet(X_train, y_train, num_clients=num_clients, alpha=dirichlet_alpha, seed=seed)
    else:
        raise ValueError("split must be 'iid' or 'dirichlet'")

    client_loaders = []
    for (Xi, yi) in client_splits:
        Xi = torch.tensor(Xi, dtype=torch.float32)
        yi = torch.tensor(yi, dtype=torch.float32)
        ds = TensorDataset(Xi, yi)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    in_dim = X_train.shape[1]
    return client_loaders, test_loader, in_dim


def _load_breast_cancer(seed=42):
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test


def _cache_path(data_dir, use_20_percent, seed):
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = "20pct" if use_20_percent else "full"
    return os.path.join(cache_dir, f"nslkdd_{tag}_seed{seed}.npz")


def _subsample_train(X_train, y_train, frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_train))
    y_int = y_train.astype(int)

    idx0 = idx[y_int == 0]
    idx1 = idx[y_int == 1]

    n0 = max(1, int(len(idx0) * frac))
    n1 = max(1, int(len(idx1) * frac))

    s0 = rng.choice(idx0, size=n0, replace=False)
    s1 = rng.choice(idx1, size=n1, replace=False)

    sel = np.concatenate([s0, s1])
    rng.shuffle(sel)

    return X_train[sel], y_train[sel]


def _load_nsl_kdd(data_dir, seed=42, use_20_percent=False):
    cache_file = _cache_path(data_dir, use_20_percent=use_20_percent, seed=seed)

    if os.path.exists(cache_file):
        print(f"[NSL-KDD] Loading cache: {os.path.abspath(cache_file)}")
        z = np.load(cache_file)
        return z["X_train"], z["y_train"], z["X_test"], z["y_test"]

    train_20_path = os.path.join(data_dir, "KDDTrain+_20Percent.txt")
    train_full_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    if not os.path.exists(train_full_path):
        raise FileNotFoundError(f"Missing train file: {train_full_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test file:  {test_path}")

    train_path = train_20_path if (use_20_percent and os.path.exists(train_20_path)) else train_full_path

    print(f"[NSL-KDD] Reading train: {os.path.abspath(train_path)}")
    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    print(f"[NSL-KDD] Reading test : {os.path.abspath(test_path)}")
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    def to_binary_label(s):
        s = str(s).strip()
        return 1.0 if s == "normal" else 0.0

    y_train = df_train["label"].apply(to_binary_label).to_numpy(dtype=np.float32)
    y_test = df_test["label"].apply(to_binary_label).to_numpy(dtype=np.float32)

    df_train = df_train.drop(columns=["label", "difficulty"])
    df_test = df_test.drop(columns=["label", "difficulty"])

    print("[NSL-KDD] One-hot encoding categorical features...")
    cat_cols = ["protocol_type", "service", "flag"]
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=cat_cols)

    X_all = df_all.to_numpy(dtype=np.float32)
    X_train = X_all[: len(df_train)]
    X_test = X_all[len(df_train):]

    print("[NSL-KDD] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if use_20_percent and not os.path.exists(train_20_path):
        print("[NSL-KDD] 20% file not found. Subsampling 20% of TRAIN set (stratified).")
        X_train, y_train = _subsample_train(X_train, y_train, frac=0.2, seed=seed)
        print(f"[NSL-KDD] Subsampled train size: {len(y_train)}")

    print(f"[NSL-KDD] Caching processed arrays to: {os.path.abspath(cache_file)}")
    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    return X_train, y_train, X_test, y_test


def get_binary_tabular_loaders(
    num_clients=3,
    batch_size=64,
    dataset="breast_cancer",
    data_dir=None,
    seed=42,
    split="iid",
    dirichlet_alpha=0.5,
    use_20_percent=False
):
    data_dir = str(DATA_DIR) if data_dir is None else str(data_dir)
    if dataset == "breast_cancer":
        X_train, y_train, X_test, y_test = _load_breast_cancer(seed=seed)
    elif dataset == "nsl_kdd":
        X_train, y_train, X_test, y_test = _load_nsl_kdd(
            data_dir=data_dir,
            seed=seed,
            use_20_percent=use_20_percent
        )
    else:
        raise ValueError("dataset must be 'breast_cancer' or 'nsl_kdd'")

    return _make_loaders_from_arrays(
        X_train, y_train, X_test, y_test,
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
        split=split,
        dirichlet_alpha=dirichlet_alpha
    )
