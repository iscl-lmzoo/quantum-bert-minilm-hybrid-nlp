
"""
QUANTUM HEAD (Qiskit Machine Learning)
- Variational Quantum Circuit (VQC) = FeatureMap + Ansatz.
- Wrapped as EstimatorQNN and exposed to PyTorch via TorchConnector.
"""
from __future__ import annotations
import torch.nn as nn
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class QuantumHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_qubits: int = 4, reps: int = 2, entanglement: str = "linear"):
        super().__init__()
        assert input_dim == num_qubits, "We set input_dim == num_qubits (the projector enforces this)."
        fm = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        ans = TwoLocal(num_qubits, rotation_blocks="ry", entanglement_blocks="cz", reps=reps, entanglement=entanglement)
        self.qnn = EstimatorQNN(circuit=fm.compose(ans), input_params=fm.parameters, weight_params=ans.parameters)
        self.conn = TorchConnector(self.qnn)
        self.out = nn.Linear(1, num_classes)  # map scalar QNN output -> logits

    def forward(self, x):
        q = self.conn(x)       # [B, 1]
        return self.out(q)     # [B, num_classes]
