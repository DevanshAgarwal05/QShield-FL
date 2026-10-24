import pennylane as qml
import torch
import torch.nn as nn


class HybridQNN(nn.Module):
    """
    Hybrid QNN with Lightning simulator.
    IMPORTANT FIX:
      - Do NOT detach feats before sending to the circuit.
      - Otherwise feature_net receives no gradient and stops learning.

    Note:
      - lightning.qubit runs on CPU.
      - If the model is on CUDA, we move feats to CPU WITHOUT detaching so gradient can flow.
-    """

    def __init__(self, in_dim=20, n_qubits=4, vqc_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.vqc_layers = vqc_layers

        self.feature_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
        )

        # ✅ Lightning backend (fast CPU quantum sim)
        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        # Keep q_weights on CPU for the quantum simulator
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
                        wires=i
                    )

                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = x.float()
        feats = self.feature_net(x)  # on CUDA if model is on CUDA

        # ✅ NO DETACH HERE — we need gradient into feature_net
        feats_cpu = feats.to("cpu")

        q_out = []
        for i in range(feats_cpu.shape[0]):
            out_list = self.circuit(feats_cpu[i], self.q_weights)
            out_vec = torch.stack(out_list)  # [n_qubits]
            q_out.append(out_vec)

        q_out = torch.stack(q_out, dim=0).float()  # [batch, n_qubits]
        q_out = q_out.to(x.device)

        logits = self.head(q_out).squeeze(-1)
        return logits
