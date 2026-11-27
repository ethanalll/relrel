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

## RT experiments

Use `make synthesize`, `make export`, or `make generate` (synthesize + export) from this repo. Set `DATA` to the same absolute path in RelRel and RT (default `~/scratch/relrel-rt`). Every run specifies hyperparameters directly—there is no `SUITE`/`ARM` indirection, and the only DAG modes are `local` and `global`.

```sh
pixi install
export DATA=/shared/scratch/relrel-rt

# Bit-exact PluRel v1.0.0 baseline (DAG=local, DECODING=original, FIXED_MLP=0, DATES=activity, EXPORT_CAT=plurel_bool)
make generate NAME=local-orig   DAG=local  DECODING=original COUNT=128 WORKERS=8
make generate NAME=local-quant  DAG=local  DECODING=quantile COUNT=128 WORKERS=8
make generate NAME=global-orig  DAG=global DECODING=original MAX_HOPS=2 COUNT=128 WORKERS=8
make generate NAME=global-quant DAG=global DECODING=quantile MAX_HOPS=2 COUNT=128 WORKERS=8

# Direct sampler matched to exact realized edges and seeds of global-quant
make generate NAME=global-direct DAG=global DECODING=quantile SAMPLER=direct \
  MATCH_EDGES_FROM=global-quant WORKERS=8

# Disjoint synthetic holdout corpus
make generate NAME=holdout-val DAG=global DECODING=quantile COUNT=16 SEED_START=100000 WORKERS=8
```

- **`DAG=local`**: Runs PluRel's per-table `SCM` directly. With default `DECODING=original`, `FIXED_MLP=0`, `DATES=activity`, and `EXPORT_CAT=plurel_bool`, output is **bit-exact** to PluRel v1.0.0 (`2a273cf`). With `DECODING=quantile`, each categorical node is eagerly discretized with Dirichlet quantile cutoffs (`ALPHA=1.0,5.0`) before descendants read it.
- **`DAG=global`**: Runs the global column-DAG RSCM (`GlobalSCMGenerator`) on the same PluRel framework (timestamps, missingness, categorical export). Controlled directly by `DECODING`, `ALPHA`, `FIXED_MLP`, `SAMPLER` (`family` | `direct`), `MAX_HOPS` (`1` | `2`), `COL_PARENT_CAP`, `DEGREE`, `SCM_FAMILIES`, `PARENT_POLICY` (`bounded` | `full`), `CELL_PARENT_CAP`, `ROW_POOLING` (`mean` | `sum`), `DATES`, `EXPORT_CAT`, `MATCH_SEEDS_FROM`, and `MATCH_EDGES_FROM`.

Raw tables and causal diagnostics go in `DATA/synthetic/raw/<NAME>/`.
Model-ready tables and the training target manifest go in `DATA/synthetic/rt/<NAME>/`.
Run preprocessing and training from the RT repo using its `EXPERIMENTS.md`.

For a smoke test, use a separate `DATA` root and pass `TINY=1 COUNT=4` to `make generate`. `make help` lists all available hyperparameter overrides.
