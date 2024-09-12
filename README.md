# scanvi-explainer

[![build](https://github.com/brickmanlab/scanvi-explainer/actions/workflows/build.yml/badge.svg)](https://github.com/brickmanlab/scanvi-explainer/actions/workflows/build.yml)

Interpretability extension for [scANVI] using [SHAP] package.

Please see our [example](Example.ipynb) notebook on how to run scANVI Explainer.

## Installation

```console
$ pip install scanvi-explainer
```

## Install from source

```console
$ git clone https://github.com/brickmanlab/scanvi-explainer.git && cd scanvi-explainer
$ uv sync
```

[scANVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html#scvi.model.SCANVI
[SHAP]: https://github.com/shap/shap

## Citation

Please consider citing scANVI Explainer if you use in your research.

> Deep Learning Based Models for Preimplantation Mouse and Human Development <br>
> Martin Proks, Nazmus Salehin, Joshua M. Brickman <br>
> bioRxiv 2024.02.16.580649; doi: [10.1101/2024.02.16.580649](https://doi.org/10.1101/2024.02.16.580649)

```BibTeX
@article{Proks2024.02.16.580649,
	author = {Proks, Martin and Salehin, Nazmus and Brickman, Joshua M.},
	title = {Deep Learning Based Models for Preimplantation Mouse and Human Development},
	elocation-id = {2024.02.16.580649},
	year = {2024},
	doi = {10.1101/2024.02.16.580649},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/02/16/2024.02.16.580649},
	eprint = {https://www.biorxiv.org/content/early/2024/02/16/2024.02.16.580649.full.pdf},
	journal = {bioRxiv}
}
```
