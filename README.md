# RelRel

Relational synthesis of relational database.

RelRel extends PluRel v1.0.0 with a column-level DAG and relational structural
causal models (RSCMs). Dependencies between cells follow foreign-key links in
either direction, with support for one- and two-hop paths.

The Python package is named `relrel`.

## Setup

Install [Pixi](https://pixi.sh), then run from this folder:

```bash
pixi install
```

## Generate data

Start with a small example:

```bash
pixi run python scripts/rscm_gen.py --tiny --count 2 --output-dir ./generated
```

Remove `--tiny` to use the original row-count and feature-count ranges.
Use `--help` to see options for seeds, FK hops, parent limits, pooling and SQL schemas.

Each database is saved under `generated/global-rscm-<seed>/`, containing Parquet
tables, generation settings, the column DAG and the FK skeleton.
Generation runs on the CPU. RT preprocessing and training are separate steps.

## Python usage

```python
from relrel import Config, SyntheticDataset

config = Config(generation_mode="global_rscm")
db = SyntheticDataset(seed=0, config=config).make_db()
```

`Config()` without this option uses the original PluRel generator.

## Tests

```bash
pixi run pytest
```

## Acknowledgments

Based on [PluRel](https://github.com/stanford-star/plurel), release v1.0.0.
See [LICENSE](LICENSE) for the license and original copyright notice.

## RT ablations

Use `make synthesize` and `make export` from this repo. Set `DATA` to the same
absolute path in RelRel and RT. The default is `~/scratch/relrel-rt`.

```sh
pixi install
export DATA=/shared/scratch/relrel-rt
make synthesize SUITE=factorial COUNT=128 WORKERS=8
make export SUITE=factorial
make synthesize SUITE=sampler COUNT=128 WORKERS=8
make export SUITE=sampler
make synthesize SUITE=holdout SPLIT=val COUNT=16 WORKERS=8
make export SUITE=holdout SPLIT=val
```

- **Factorial** has four arms combining per-table SCMs or global RSCM with
  original categorical decoding or eager quantile decoding.
- **Sampler** has two global-RSCM arms with eager quantiles. For each seed, the
  direct sampler matches the original graph-family sampler's realized edge
  count. Pairs that cannot meet the budget are recorded and replaced.
- **Holdout** creates one reference corpus using a mixture of the four factorial
  settings and starting from a separate seed range.

Each accepted pair or quartet shares the schema, FK skeleton, feature
inventory, missingness draws, and eligible RT targets. Both generator arms
produce static tables. The per-table SCM retains latent nodes and dense
forward-FK inputs, while the global SCM uses its column-level edges. The
factorial therefore measures the effect of switching between these generator
pipelines. Both quantile arms discretize each categorical column before its
descendants read it.

Raw tables and causal diagnostics go in `DATA/synthetic/raw/<suite>/<split>/`.
Model-ready tables and the training target manifest go in
`DATA/synthetic/rt/<suite>/<split>/<arm>/`. The export represents multicategory
features as text contexts and preserves Boolean and numerical training targets.
Run preprocessing and training from the RT repo using its `EXPERIMENTS.md`.

For a smoke test, use a separate `DATA` root and pass `TINY=1 COUNT=4` to
`make synthesize`. `make help` lists the available overrides.
