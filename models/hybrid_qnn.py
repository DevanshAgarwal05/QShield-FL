import torch
import torch.nn as nn

from models.circuit import build_circuit


class HybridQNN(nn.Module):
    """Classical feature net -> VQC (Lightning simulator) -> linear head.

    Used for both the raw-feature baseline and the SVD path; for SVD, `in_dim`
    is the projected rank and the projection happens outside the model.

    The simulator runs on CPU. Features are moved to CPU without detaching so
    gradients still reach feature_net.
    """

    def __init__(self, in_dim=20, n_qubits=4, vqc_layers=3, hidden_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.vqc_layers = vqc_layers
        self.hidden_dim = hidden_dim

        self.feature_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits),
        )
        self.q_weights = nn.Parameter(0.01 * torch.randn(vqc_layers, n_qubits, 3, device="cpu"))
        self.circuit = build_circuit(n_qubits, vqc_layers)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        feats_cpu = self.feature_net(x.float()).to("cpu")

        q_out = torch.stack(
            [torch.stack(self.circuit(feats_cpu[i], self.q_weights)) for i in range(feats_cpu.shape[0])],
            dim=0,
        ).float().to(x.device)

        return self.head(q_out).squeeze(-1)
