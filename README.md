# scanvi-explainer

[![build][build-badge]][build-link]
[![Documentation Status][docs]][docs-link]

Interpretability extension for [scANVI] using [SHAP] package.

Please see our [example](docs/notebooks/Example.ipynb) notebook on how to run scANVI Explainer.

## Installation

```console
$ pip install scanvi-explainer
```

## Install from source

```console
$ git clone https://github.com/brickmanlab/scanvi-explainer.git && cd scanvi-explainer
$ uv sync
```

## Build documentation

```console
$ sphinx-build -M html docs docs/_build
```

## Citation

Please consider citing scANVI Explainer if you use in your research.

> Deep Learning Based Models for Preimplantation Mouse and Human Development <br>
> Martin Proks, Nazmus Salehin, Joshua M. Brickman <br>
> bioRxiv 2024.02.16.580649; doi: [10.1101/2024.02.16.580649]

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

[docs]: https://readthedocs.org/projects/scanvi-explainer/badge/?version=latest
[docs-link]: https://scanvi-explainer.readthedocs.io/en/latest/?badge=latest
[build-badge]: https://github.com/brickmanlab/scanvi-explainer/actions/workflows/build.yml/badge.svg
[build-link]: https://github.com/brickmanlab/scanvi-explainer/actions/workflows/build.yml
[scANVI]: https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html#scvi.model.SCANVI
[SHAP]: https://github.com/shap/shap
[10.1101/2024.02.16.580649]: https://doi.org/10.1101/2024.02.16.580649
