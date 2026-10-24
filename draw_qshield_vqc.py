import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pennylane as qml
import torch


def build_qnode(n_qubits: int = 4, vqc_layers: int = 3):
    """Build the exact VQC topology used in hybrid_qnn.py/hybrid_svd_qnn.py."""
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        # Fallback only for drawing in environments without lightning plugin.
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(x, weights):
        # Exact embedding from project models.
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

        # Exact variational and entanglement pattern from project models.
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

        # Exact readout structure from project models.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def draw_and_save(output_path: Path, n_qubits: int = 4, vqc_layers: int = 3, dpi: int = 300):
    circuit = build_qnode(n_qubits=n_qubits, vqc_layers=vqc_layers)

    # Safe dummy values for static visualization only.
    x = torch.linspace(0.1, 0.4, steps=n_qubits, dtype=torch.float32)
    weights = 0.01 * torch.randn(vqc_layers, n_qubits, 3, dtype=torch.float32)

    fig, _ = qml.draw_mpl(circuit, style="black_white")(x, weights)
    fig.set_size_inches(16, 6)

    # Report-friendly annotations clarifying what the reader is seeing.
    fig.suptitle(
        f"QSHIELD-FL Variational Quantum Circuit (n_qubits={n_qubits}, layers={vqc_layers})",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Embedding: AngleEmbedding(rotation='Y'); per-layer gates: Rot x n_qubits + ring CNOT(i -> i+1 mod n); readout: expval(PauliZ) on all qubits.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("Circuit matched from: models/hybrid_qnn.py::HybridQNN.__init__->circuit and")
    print("                      models/hybrid_svd_qnn.py::HybridSVDQNN.__init__->circuit")
    print(f"Image created: {output_path.name}")
    print(f"Saved at: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Draw QSHIELD-FL VQC topology to image.")
    parser.add_argument("--output", default="qshield_vqc.png", help="Output image filename/path.")
    parser.add_argument("--n_qubits", type=int, default=4, help="Qubit count (project default: 4).")
    parser.add_argument("--vqc_layers", type=int, default=3, help="VQC layer count (project default: 3).")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    args = parser.parse_args()

    draw_and_save(
        output_path=Path(args.output),
        n_qubits=args.n_qubits,
        vqc_layers=args.vqc_layers,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
