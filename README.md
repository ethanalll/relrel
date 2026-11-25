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

Each accepted pair or quartet shares the schema, FK skeleton, feature inventory,
activity dates, missingness draws, and eligible RT targets. The table schema uses
the three original schema families. Local SCMs draw independently per table
from the five original SCM families. The global SCM draws one database-wide
proposal from those same five families and filters edges by FK reachability.
The per-table SCM retains latent nodes and dense forward-FK inputs. Both
quantile arms discretize each categorical column before descendants read it.

Use a fresh `DATA` root for the dated paired corpora. Their manifests record the
`paired-activity-dates-v1` generation profile. For exact historical references,
use the [original PluRel v1.0.0 generator](https://github.com/stanford-star/plurel/tree/2a273cfd21933ee4893dfcb862a3edaed45ac665)
and the [RelRel 40cca23 generator](https://github.com/ethanalll/relrel/tree/40cca23880063cb0d74829ef387e01a028600563).
The original PluRel release generates dated activity tables and samples MLP
activations per call. RelRel 40cca23 exports timestamp-free tables and fixes
MLP activations. It also retains multiclass categorical values. The paired
factorial uses dates, fixed activations, and a common export policy across all
four arms. To generate the two standalone references under the same `DATA` root:

```sh
make reference REFERENCE=plurel COUNT=4 SEED_START=10000
make reference REFERENCE=relrel COUNT=4 SEED_START=10000
```

These commands write to `DATA/synthetic/reference/<reference>/`. The RelRel
reference uses quantile decoding with the five-family prior. For the PluRel
Table 1 comparison, use the
[released synthetic-only checkpoint](https://huggingface.co/stanford-star/rt-plurel/blob/main/paper/synthetic-pretrain_rdb_1024_size_4b.pt).

Raw tables and causal diagnostics go in `DATA/synthetic/raw/<suite>/<split>/`.
Model-ready tables and the training target manifest go in
`DATA/synthetic/rt/<suite>/<split>/<arm>/`. The export represents multicategory
features as text contexts and preserves Boolean and numerical training targets.
Run preprocessing and training from the RT repo using its `EXPERIMENTS.md`.

For a smoke test, use a separate `DATA` root and pass `TINY=1 COUNT=4` to
`make synthesize`. `make help` lists the available overrides.
