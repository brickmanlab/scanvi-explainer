from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scvi.model import SCANVI
from tqdm.auto import tqdm

from scanvi_explainer.scanvi_deep import SCANVIDeep

from .utils import get_labels_key


class SCANVIBoostrapper:
    def __init__(self, model: SCANVI, n_bootstraps: int = 25):
        """Bootrap wrapper

        Parameters
        ----------
        model
            Trained :class:`~scvi.model.SCANVI` model
        n_bootstraps : int
            Number of bootstraps, by default 25
        """

        self.model = model
        self.NUM_OF_BOOTSRAPS = n_bootstraps

    def run(self, train_size: float = 0.8, batch_size: int = 128) -> list[np.ndarray]:
        """Runner

        Parameters
        ----------
        train_size
            :obj:`float` Training size (background), by default 0.8
        batch_size
            :obj:`int` Number of cells used from each group, by default 128
            To ignore the batch_size subsetting, set batch_size=-1

        Returns
        -------
        list[np.ndarray]
            List of SHAP bootstrap values
        """

        values = [
            SCANVIDeep(self.model, train_size, batch_size).shap_values(with_labels=True)
            for _ in tqdm(range(self.NUM_OF_BOOTSRAPS))
        ]

        return values

    def estimate(
        self,
        func_fn: Callable[..., ArrayLike],
        shap_values: list[ArrayLike],
    ) -> np.ndarray:
        """Calculate measurement for each boostrap run.

        Parameters
        ----------
        func_fn : Callable[..., ArrayLike]
            Stat function to call, i.e.: np.mean, np.median. The function must contain axis parameter.
        shap_values : list[ArrayLike]
            SHAP values

        Returns
        -------
        np.ndarray
            Array of metric in format (classifier code, n_features, n_bootstraps)
        """

        if not shap_values:
            raise ValueError(
                "Empty SHAP values, make sure the list contains at least one bootstrap!"
            )

        n_labels, n_test, n_features = shap_values[0][1].shape
        res = np.zeros((n_labels, n_features, self.NUM_OF_BOOTSRAPS))
        label_codes = range(n_labels)

        for idx in label_codes:
            res[idx] = np.vstack(
                [
                    func_fn(shaps[idx, (mask == idx).ravel(), :], axis=0)
                    for mask, shaps in shap_values
                ]
            ).T

        return res

    @staticmethod
    def _filter(
        shap_values: list, features: list[str], top_n: None | int = 10
    ) -> pd.DataFrame:
        """Helper function for filtering top positive only SHAP features.

        Parameters
        ----------
        shap_values : list
            SHAP values
        features : list[str]
            Features (genes)
        top_n : int
            Number of top features to subset, by default 10. If None is specified, the filter does
            not apply.

        Returns
        -------
        pd.DataFrame
            Filtered dataset in DataFrame format
        """
        data = pd.DataFrame(shap_values, index=features)
        data = data[data > 0.0].dropna()
        order = data.mean(axis=1).sort_values(ascending=False).head(top_n).index

        return data.loc[order]

    def feature_plot(
        self,
        shap_values: list,
        n_features: int = 10,
        metric_fn: Callable[..., ArrayLike] = np.mean,
        kind: Literal["boxplot", "barplot"] = "boxplot",
        gene_symbols: None | str = None,
        n_cols: int = 3,
        figsize: tuple[int, int] = (20, 20),
        return_fig: bool = False,
    ) -> Figure | None:
        """Feature plot for bootstrapping

        Parameters
        ----------
        shap_values : list
            SHAP values
        n_features : int
            Number of features to subset, by default 10
        metric : Callable[..., ArrayLike]
            Statistical measurement of each boostrap, by default np.mean
        kind : Literal[&quot;boxplot&quot;, &quot;barplot&quot;]
            Type of plot, by default "boxplot"
        gene_symbols: None | str = None
            Column name in `var` for gene symbols
        n_cols : int
            Number of columns for subplots, by default 3
        figsize : tuple[int, int]
            Figure size, by default [20, 20]
        return_fig : bool
            Flag to return figure object, by default False

        Returns
        -------
        Figure | None
            Either plot or return figure object

        Raises
        ------
        ValueError
            When specified plot kind is not supported
        """

        if kind not in ["boxplot", "barplot"]:
            raise ValueError(f"Specified {kind} not supported!")

        if gene_symbols and gene_symbols not in self.model.adata.var.columns:
            raise ValueError(
                "Specified gene_symbol not present in the 'var' of model's adata!"
            )

        features = (
            self.model.adata.var[gene_symbols]
            if gene_symbols
            else self.model.adata.var_names
        )
        sample_stat = self.estimate(metric_fn, shap_values)
        labels = self.model.adata.obs[get_labels_key(self.model)].cat.categories

        n_rows = round(labels.size / n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        for idx, label in enumerate(labels):
            data = self._filter(sample_stat[idx], features, n_features).T

            if kind == "boxplot":
                sns.boxplot(
                    orient="h",
                    ax=ax[idx // n_cols, idx % n_cols],
                    data=data,
                ).set(title=label)
                sns.stripplot(
                    orient="h",
                    color="black",
                    alpha=0.4,
                    ax=ax[idx // n_cols, idx % n_cols],
                    data=data,
                )
            if kind == "barplot":
                sns.barplot(
                    orient="h",
                    errorbar=("ci", 95),
                    capsize=0.2,
                    ax=ax[idx // n_cols, idx % n_cols],
                    data=data,
                ).set(title=label)

        # clean axes which are empty
        # from: https://stackoverflow.com/a/76269136
        _ = [fig.delaxes(ax_) for ax_ in ax.flatten() if not ax_.has_data()]

        fig.suptitle(
            f"Top {n_features} SHAP values based on boostrapping (n={self.NUM_OF_BOOTSRAPS})"
        )
        fig.supxlabel("SHAP values")
        fig.supylabel("Features (genes)")
        fig.tight_layout()

        if return_fig:
            return fig

        return None

    def save(self, shap_values: list[np.ndarray], filename: str):
        """Save results to feather format.

        Parameters
        ----------
        shap_values : list[np.ndarray]
            SHAP values
        filename : str
            Path to filename containing .feather extension

        Examples
        --------
        >>> lvae = scvi.model.SCANVI.load("...")
        >>> bootstrapper = SCANVIBoostrapper(lvae, n_bootstraps=10)
        >>> shap_values = bootstrapper.run(train_size=0.8, batch_size=64)
        >>> bootstrapper.save(shap_values, "./bootstrapped_shaps.feather")
        """

        features = self.model.adata.var_names
        sample_stat = self.estimate(np.mean, shap_values)
        labels = self.model.adata.obs[get_labels_key(self.model)].cat.categories

        res = []
        for idx, label in enumerate(labels):
            data = self._filter(sample_stat[idx], features, top_n=None)
            data.columns = [f"n_{c}" for c in data.columns]
            data["label"] = label

            res.append(data)

        if not filename.endswith(".feather"):
            filename += ".feather"

        try:
            pd.concat(res).reset_index().to_feather(filename)
        except IOError:
            raise
