```{toctree}
:hidden: true
:maxdepth: 1

install.md
api.md
notebooks/Example
references.md
changelog.md
GitHub <https://github.com/brickmanlab/scanvi-explainer>
AI Models <https://huggingface.co/brickmanlab>
```

# Documentation

```{eval-rst}
`scanvi-explainer` is a package which helps to describe which features (genes)
are required when predicting a cell type using SCANVI :cite:p:`Xu2021`. 
We utilize SHAP package :cite:p:`NIPS2017_7062` to estimate the importance of 
each gene.

This package follows best practices from `scvi-tools` :cite:p:`Gayoso2022`.
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

[10.1101/2024.02.16.580649]: https://doi.org/10.1101/2024.02.16.580649