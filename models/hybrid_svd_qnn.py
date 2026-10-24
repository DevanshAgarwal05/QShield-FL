import pennylane as qml
import torch
import torch.nn as nn


class HybridSVDQNN(nn.Module):
    """
    SVD-enhanced hybrid QNN path.

    Expected pipeline:
      preprocessed tabular features -> optional SVD projection (outside model)
      -> classical feature extractor -> VQC -> classical output head

    This class is intentionally separate from `HybridQNN` so baseline behavior
    remains unchanged.
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

        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        q_init = 0.01 * torch.randn(vqc_layers, n_qubits, 3, device="cpu")
        self.q_weights = nn.Parameter(q_init)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

            for l in range(vqc_layers):
                for i in range(n_qubits):
                    qml.Rot(
                        weights[l, i, 0],
                        weights[l, i, 1],
                        weights[l, i, 2],
                        wires=i,
                    )

                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = x.float()
        feats = self.feature_net(x)
        feats_cpu = feats.to("cpu")

        q_out = []
        for i in range(feats_cpu.shape[0]):
            out_list = self.circuit(feats_cpu[i], self.q_weights)
            out_vec = torch.stack(out_list)
            q_out.append(out_vec)

        q_out = torch.stack(q_out, dim=0).float().to(x.device)
        logits = self.head(q_out).squeeze(-1)
        return logits
