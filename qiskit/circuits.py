# src/quantum/circuits.py
"""
Quantum circuit builders and QNN factory:
- build_feature_map: ZZFeatureMap with angle encoding
- build_ansatz: EfficientSU2 layered ansatz
- make_estimator_qnn: EstimatorQNN wrapped for classification
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN

def build_feature_map(num_qubits: int, reps: int = 2) -> ZZFeatureMap:
    """Angle-encoding feature map with entanglement."""
    return ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement="linear")

def build_ansatz(num_qubits: int, reps: int = 2) -> EfficientSU2:
    """Hardware-efficient ansatz with tunable depth."""
    return EfficientSU2(num_qubits=num_qubits, reps=reps, entanglement="linear")

def make_estimator_qnn(num_qubits: int,
                       feature_map: Optional[ZZFeatureMap] = None,
                       ansatz: Optional[EfficientSU2] = None) -> EstimatorQNN:
    """
    Compose feature_map + ansatz into an EstimatorQNN suitable for
    qiskit_machine_learning.algorithms.NeuralNetworkClassifier.
    """
    if feature_map is None:
        feature_map = build_feature_map(num_qubits)
    if ansatz is None:
        ansatz = build_ansatz(num_qubits)

    # EstimatorQNN will internally build <Z> expectation on the last qubit by default
    qnn = EstimatorQNN(
        circuit=ansatz.compose(feature_map, front=True),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    return qnn
