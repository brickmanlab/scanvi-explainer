import logging

import anndata
import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.model import SCANVI
from sklearn.model_selection import train_test_split


def get_labels_key(model: SCANVI) -> str:
    """Retrieve key from `.obs` with labels used to train the classifier

    Parameters
    ----------
    model
        :class:`~scvi.model.SCANVI` model

    Returns
    -------
    str
        labels key
    """
    return model.registry_["setup_args"]["labels_key"]


def get_layer_key(model: SCANVI) -> str:
    """Retrieve layers key of counts which the model was trained on

    Parameters
    ----------
    model
        :class:`~scvi.model.SCANVI` model

    Returns
    -------
    str
        layers key
    """
    return model.registry_["setup_args"]["layer"]


def train_test_group_split(
    adata: anndata.AnnData,
    groupby: str,
    train_size: float = 0.8,
    batch_size: int = 128,
    layer: str = "counts",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Function to split anndata object 80/20 per group in format required for SCANVIDeep explainer.

    Bigger datasets might not fit to the GPU memory. To overcome this issue we recommend setting
    `batch_size` to 128, meaning each group will only use randomly sampled 128 cells. This speeds
    up the explainer at the cost of correctness. We suggest bootstrapping this process multiple
    times.

    Parameters
    ----------
    adata
        :class:`~anndata.AnnData` Annotated dataset
    groupby : str
        Column in metadata by which the dataset should be split by
    train_size
        :obj:`float` Training size (background), by default 0.8
    batch_size
        :obj:`int` Number of cells used from each group, by default 128

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        Train and test splits
    """

    if groupby not in adata.obs.columns:
        raise ValueError(f"Provided group {groupby} not found in `.obs` metadata!")

    if 0.0 < train_size > 1.0:
        raise ValueError("Training size has to be between 0 to 1!")

    rng = np.random.default_rng()
    test_size = 1.0 - train_size
    groups = adata.obs.groupby(groupby)

    train, test = [], []
    for _, cells in groups.groups.items():
        cells = (
            rng.choice(cells.values, batch_size, replace=False)
            if len(cells.values) > batch_size and batch_size != -1
            else cells.values
        )
        train_split, test_split = train_test_split(cells, test_size=test_size)

        train.append(train_split)
        test.append(test_split)
        logging.debug(f"{cells.shape} / {train_split.shape} / {test_split.shape}")

    train, test = np.concatenate(train), np.concatenate(test)

    X_train = {
        REGISTRY_KEYS.X_KEY: torch.from_numpy(
            adata[train].layers[layer].todense()
        ).type(torch.float32),
        REGISTRY_KEYS.BATCH_KEY: torch.from_numpy(
            adata[train]
            .obs[REGISTRY_KEYS.BATCH_KEY]
            .cat.codes.values[:, np.newaxis]
            .copy()
        ),
        REGISTRY_KEYS.LABELS_KEY: torch.from_numpy(
            adata[train].obs[groupby].cat.codes.values[:, np.newaxis].copy()
        ),
    }

    X_test = {
        REGISTRY_KEYS.X_KEY: torch.from_numpy(adata[test].layers[layer].todense()).type(
            torch.float32
        ),
        REGISTRY_KEYS.BATCH_KEY: torch.from_numpy(
            adata[test]
            .obs[REGISTRY_KEYS.BATCH_KEY]
            .cat.codes.values[:, np.newaxis]
            .copy()
        ),
        REGISTRY_KEYS.LABELS_KEY: torch.from_numpy(
            adata[test].obs[groupby].cat.codes.values[:, np.newaxis].copy()
        ),
    }

    return (X_train, X_test)
