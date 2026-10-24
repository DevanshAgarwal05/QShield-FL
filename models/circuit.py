import pennylane as qml


def build_circuit(n_qubits=4, vqc_layers=3, device_name="lightning.qubit"):
    """Return the VQC QNode shared by the model and the circuit drawing.

    AngleEmbedding(Y) -> vqc_layers x (Rot per qubit + ring CNOT) -> <Z> on every qubit.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

        for l in range(vqc_layers):
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit
