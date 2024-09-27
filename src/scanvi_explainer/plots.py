import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scvi import REGISTRY_KEYS

from .scanvi_deep import SCANVIDeep


def feature_plot(
    explainer: SCANVIDeep,
    shap_values: np.ndarray,
    subset: bool = False,
    top_n: int = 10,
    gene_symbols: None | str = None,
    n_cols: int = 2,
    figsize: tuple[int, int] = (20, 20),
    return_fig: bool = False,
) -> Figure | None:
    """Prints feature contribution (absolute mean SHAP value) for each cell type (top 10).

    Parameters
    ----------
    explainer
        :class:`~scanvi_explainer.scanvi_deep.SCANVIDeep` explainer
    shap_values
        :class:`~numpy.ndarray` Expected SHAP values
    subset
        When set to true, calculate contribution by subsetting for test cells which belong to that
        particual classifier.
        When set to false, be generic and return contributing features even when testing set has
        different cell types.
    top_n: int
        Subset for top N number of features
    gene_symbols: None | str = None
        Column name in `var` for gene symbols
    n_cols: int
        Number of columns in Figure
    figsize : tuple[int, int]
            Figure size, by default [20, 20]
    return_fig : bool
        Flag to return figure object, by default False
    """

    if gene_symbols and gene_symbols not in explainer.adata.var.columns:
        raise ValueError(
            "Specified gene_symbol not present in the 'var' of model's adata!"
        )

    groupby = explainer.labels_key
    classes = explainer.adata.obs[groupby].cat.categories
    features = (
        explainer.adata.var[gene_symbols].values
        if gene_symbols
        else explainer.adata.var_names
    )

    nrows = round(classes.size / n_cols)
    fig, ax = plt.subplots(nrows, n_cols, sharex=False, figsize=figsize)

    for idx, ct in enumerate(classes):
        shaps = pd.DataFrame(shap_values[idx], columns=features)

        if subset:
            shaps[groupby] = explainer.test[REGISTRY_KEYS.LABELS_KEY]
            shaps = shaps[shaps[groupby] == idx].iloc[:, :-1]

            tmp_avg = (
                shaps.mean(axis=0)
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "feature", 0: "weight"})
            )
            positive = (
                tmp_avg.query("weight > 0")
                .head(top_n // 2)
                .assign(contribution="positive")
            )
            negative = (
                tmp_avg.query("weight < 0")
                .tail(top_n // 2)
                .assign(contribution="negative")
            )

            avg = pd.concat([positive, negative])
            title = f"Mean(|SHAP value|) importance for: {ct}"

        else:
            avg = (
                shaps.abs()
                .mean(axis=0)
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "feature", 0: "weight"})
                .query("weight > 0")
                .head(10)
            )
            title = f"Mean(|SHAP value|) importance for: {ct}"

        sns.barplot(
            x="weight",
            y="feature",
            hue="contribution",
            palette=["red", "blue"],
            data=avg,
            ax=ax[idx // n_cols, idx % n_cols],
        )

        ax[idx // n_cols, idx % n_cols].set_title(title)
        ax[idx // n_cols, idx % n_cols].legend(title="Contribution", loc="lower right")

    # clean axes which are empty
    # from: https://stackoverflow.com/a/76269136
    _ = [fig.delaxes(ax_) for ax_ in ax.flatten() if not ax_.has_data()]

    fig.tight_layout()

    if return_fig:
        return fig

    return None
