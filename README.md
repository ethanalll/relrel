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
