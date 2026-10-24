"""SVD (PCA-style) projection applied to the federated loaders."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def fit_svd_projection(client_loaders, rank):
    x_all = torch.cat([dl.dataset.tensors[0].detach().cpu().float() for dl in client_loaders], dim=0)
    rank = int(max(1, min(rank, x_all.shape[1])))

    mean_vec = x_all.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x_all - mean_vec, full_matrices=False)
    components = vh[:rank, :].T.contiguous()  # [d, rank]
    return mean_vec.squeeze(0), components


def project(x, mean_vec, components):
    return (x - mean_vec) @ components


def apply_svd_to_loaders(client_loaders, test_loader, rank, batch_size):
    """Fit on the concatenated client data; return projected loaders, new dim, mean, components."""
    mean_vec, components = fit_svd_projection(client_loaders, rank)
    mean_vec, components = mean_vec.float(), components.float()

    new_client_loaders = []
    for dl in client_loaders:
        x, y = dl.dataset.tensors
        ds = TensorDataset(project(x.float(), mean_vec, components), y.float())
        new_client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    x_test, y_test = test_loader.dataset.tensors
    new_test_loader = DataLoader(
        TensorDataset(project(x_test.float(), mean_vec, components), y_test.float()),
        batch_size=batch_size,
        shuffle=False,
    )
    return new_client_loaders, new_test_loader, int(components.shape[1]), mean_vec, components
