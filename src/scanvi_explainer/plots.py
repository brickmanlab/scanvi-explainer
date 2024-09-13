import matplotlib.pyplot as plt
import numpy.typing as ntp
import pandas as pd
import seaborn as sns
from scvi import REGISTRY_KEYS

from scanvi_explainer.scanvi_deep import SCANVIDeep


def feature_plot(
    explainer: SCANVIDeep,
    shap_values: ntp.ArrayLike,
    subset: bool = False,
    top_n: int = 10,
) -> None:
    """Prints feature contribution (absolute mean SHAP value) for each cell type (top 10).

    Parameters
    ----------
    explainer
        :class:`~scanvi_explainer.scanvi_deep.SCANVIDeep` explainer
    shap_values
        :class:`numpy.typing.ArrayLike` Expected SHAP values
    subset : :obj:`bool`, optional
        When set to true, calculate contribution by subsetting for test cells which belong to that
        particual classifier.
        When set to false, be generic and return contributing features even when testing set has
        different cell types.
    """

    groupby = explainer.labels_key
    classes = explainer.adata.obs[groupby].cat.categories
    features = explainer.adata.var_names

    nrows = classes.size // 2 + classes.size % 2
    fig, ax = plt.subplots(nrows, 2, sharex=False, figsize=[20, 40])

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
            title = f"Mean(SHAP value average importance for: {ct}"

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
            title = f"Mean(|SHAP value|) average importance for: {ct}"

        sns.barplot(
            x="weight",
            y="feature",
            hue="contribution",
            palette=["red", "blue"],
            data=avg,
            ax=ax[idx // 2, idx % 2],
        )
        ax[idx // 2, idx % 2].set_title(title)
        ax[idx // 2, idx % 2].legend(title="Contribution", loc="lower right")
        fig.tight_layout()
