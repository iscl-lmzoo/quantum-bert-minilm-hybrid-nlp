"""Top-level package for the Qiskit + LLM hybrid demo.

Exposes a small, convenient public API and package metadata.
"""
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFoundError

__all__ = [
    "embed_texts",
    "load_csv",
    "build_feature_map",
    "build_ansatz",
    "train",
    "evaluate",
    "__version__",
]

# Package version (best effort). Falls back to a static string if not installed as a package.
try:
    __version__ = _pkg_version("qiskit-llm-hybrid")  # if packaged & installed
except _PkgNotFoundError:
    __version__ = "0.1.0"

# Re-export common entrypoints
from .embeddings import embed_texts
from .data_loader import load_csv
from ..qiskit.circuits import build_feature_map, build_ansatz
from .train_hybrid import train
from .evaluate import evaluate
