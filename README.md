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

To enable the categorical and column-degree priors together:

```bash
pixi run python scripts/rscm_gen.py --tiny --count 5 --output-dir ./generated-priors \
  --categorical-alpha-choices 1 5 --column-degree-choices 2 4 8
```

For each categorical column, the generator draws a symmetric Dirichlet
concentration uniformly from the supplied choices. It fits scalar cutoffs on
the baseline scores, keeps tied scores together, and reuses the cutoffs under
interventions. The causal sidecar
records the target proportions, cutoffs, and realized baseline counts. Tied
scores and rounding in small tables can leave some categories empty.

For each database, the degree prior samples a target average total degree
`d = 2 * edges / columns`. The sampler rounds the implied edge count
stochastically and samples edges from a random column order, respecting the
FK hop limit and column parent cap. It uses all available edges under those
constraints if the requested count is infeasible. The graph sidecar records
both the target and the realized degree.

Every generated feature is exported, including constant columns. Binary
features are booleans, multiclass integer features retain their category IDs,
and SQL string labels are strings. Use a fresh output directory when
regenerating databases produced by an older export format.

Remove `--tiny` to use the original row-count and feature-count ranges.
Use `--help` to see options for seeds, FK hops, parent limits, pooling and SQL schemas.

Each database is saved under `generated/global-rscm-<seed>/`, containing Parquet
tables, generation settings, the column DAG and the FK skeleton.
Generation runs on the CPU. RT preprocessing and training are separate steps.

## Python usage

```python
from relrel import Config, SyntheticDataset

config = Config()
db = SyntheticDataset(seed=0, config=config).make_db()
```

## Export for RT `rt-v1`

Generate an RT-ready copy alongside the native RelRel cache:

```bash
pixi run python scripts/rscm_gen.py --tiny --output-dir ./generated \
  --categorical-alpha-choices 1 5 --column-degree-choices 2 4 8 \
  --rt-v1-export-dir ~/scratch/relbench
```

The RT copy preserves the generated columns. It represents multiclass IDs and
constant Booleans as text context, since RT v1 treats integer IDs as numbers and
divides constant Boolean values by zero. Its `train_tasks.json` lists varying
numerical and Boolean targets in RT's four-field tuple format. Random schemas
generated for this export have at most five foreign keys per table. A supplied
SQL schema must meet the same limit.

From the unmodified RT checkout, run:

```bash
(cd rustler && pixi run cargo run --release -- pre global-rscm-0)
pixi run python -m rt.embed global-rscm-0
```

Load `~/scratch/relbench/global-rscm-0/train_tasks.json` and pass its tuples
as `train_tasks` to `rt.main.main`. Use `eval_tasks=[]` for a training-only
run on the synthetic data. RT's original evaluator can still use the paper's
RelBench tasks. Use a fresh export directory or a different seed offset to
avoid replacing an existing RT dataset.

## Tests

```bash
pixi run pytest
```

## Acknowledgments

Based on [PluRel](https://github.com/stanford-star/plurel), release v1.0.0.
See [LICENSE](LICENSE) for the license and original copyright notice.
