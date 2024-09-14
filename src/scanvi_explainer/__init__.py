import importlib.util

from .scanvi_deep import SCANVIDeep

if not importlib.util.find_spec("torch"):
    raise ImportError("Missing torch package! Run pip install torch")

if not importlib.util.find_spec("scvi"):
    raise ImportError("Missing torch package! Run pip install scvi-tools")

__all__ = ["SCANVIDeep", "utils", "plots"]
