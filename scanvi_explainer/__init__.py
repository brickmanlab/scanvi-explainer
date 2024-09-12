from .scanvi_deep import SCANVIDeep

try:
    import torch
except ImportError:
    raise ImportError("Missing torch package! Run pip install torch")

try:
    from scvi.model import SCANVI
except ImportError:
    raise ImportError("Missing scvi-tools package! Run pip install scvi-tools")

__all__ = [
    "SCANVIDeep"
]