
"""
HYBRID MODELS
- BertEncoder -> [CLS]
- Projector: CLS -> num_qubits scaled to [-pi, pi]
- Head: classical linear OR quantum (EstimatorQNN)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from quantum_layer import QuantumHead

class BertEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", fine_tune: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS

class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return torch.tanh(self.lin(x)) * math.pi  # scale to [-pi, pi]

class ClassicalHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x): return self.fc(x)

class HybridModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, head: str = 'quantum',
                 fine_tune_bert: bool = False, num_qubits: int = 4, vqc_reps: int = 2):
        super().__init__()
        self.enc = BertEncoder(model_name=model_name, fine_tune=fine_tune_bert)
        hidden = self.enc.config.hidden_size
        self.head_type = head
        if head == 'classical':
            self.classical = ClassicalHead(hidden, num_classes)
            self.projector = None
        else:
            self.projector = Projector(hidden, num_qubits)
            self.quantum = QuantumHead(num_qubits, num_classes, num_qubits=num_qubits, reps=vqc_reps)

    def forward(self, input_ids, attention_mask):
        cls = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        if self.head_type == 'classical':
            return self.classical(cls)
        x = self.projector(cls)
        return self.quantum(x)
