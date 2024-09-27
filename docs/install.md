# Installation

```console
$ pip install scanvi-explainer
```

## Install from source

```console
$ git clone https://github.com/brickmanlab/scanvi-explainer.git && cd scanvi-explainer
$ uv sync --extra dev --extra doc
```

## Build documentation

At the moment, `uv` does not support runners, so in the meantime, manual execution
is required.

```console
$ rm -rf docs/{_build,generated} && sphinx-build -M html docs docs/_build
```
