# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixed

- Methods were missing detailed explaination in the documentation

## Version 0.3.1

### Added

- `SCANVIBoostrapper.save` for saving bootstrap results into feather format [0ebd757]
- New parameter `gene_symbols` for `plots.feature_plot` and `SCANVIBoostrapper.feature_plot` [81349a7]

### Changed

- Empty subplots are removed from the figure [81349a7]

## Version 0.3.0

### Added

- `SCANVIBoostrapper` for automatic bootstrapping [bb7d12b](https://github.com/brickmanlab/scanvi-explainer/pull/4/commits/bb7d12bb99f85ac832a0fc373383513e310ae16e)
- `mypy` linting as part of CI [236bb75](https://github.com/brickmanlab/scanvi-explainer/pull/4/commits/236bb75db072bffad52fa58756cd0ef071ed6116)

### Fixed

- Readonly warning when creating train and test `Tensor` objects [4442efc](https://github.com/brickmanlab/scanvi-explainer/pull/4/commits/4442efc81235807c8d778549e2c025c9d75b0018)


## Version 0.2.0

> [!WARNING]  
> **Breaking changes**
> New changes to the API, please see the documentation

### Changed

- new api documentation
- new api changes
- use uvx for running tools

## Version 0.1.0

**NOTE**: Use this version with initial release of the AI models `v1.0`.

Initial release of [biorXiv](https://www.biorxiv.org/content/10.1101/2024.02.16.580649v1)
version as a python package.