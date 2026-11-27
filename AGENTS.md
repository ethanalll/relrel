# RelRel (`relrel/`)

Synthetic relational database generator extending PluRel v1.0.0 with a global feature-column DAG, Relational Structural Causal Models (RSCMs), and paired ablation corpus generation for Relational Transformer (RT) experiments.

## Scope & Repository Boundaries

- **This repo owns**: synthetic database generation (`relrel/`) and paired corpus synthesis/export into `DATA/synthetic/raw/` and `DATA/synthetic/rt/`.
- **Do not use `relrel/rt/` or `relrel/rustler/` for active experiments**: those are historical PluRel v1.0.0 remnants kept only for legacy compatibility/CI. All active RT preprocessing, training, evaluation, and summarization live in the sibling `../relational-transformer/` repository.
- Pass the **same absolute `DATA` root** (default `~/scratch/relrel-rt`) to Make targets in both `relrel/` and `relational-transformer/`.

## Module Map

| Path | Responsibility |
| --- | --- |
| `relrel/config.py` | Frozen dataclasses: `Choices`, `DatabaseParams`, `SCMParams`, `DAGParams`, `RSCMParams`, `Config` (`generation_mode`: `"legacy_scm"` or `"global_rscm"`). |
| `relrel/dataset.py` | `SyntheticDataset(relbench.base.Dataset)`. Dispatches `make_db()` to legacy PluRel or `_make_global_db()`, manages RelBench cache validation (`generation.json`, `causal.json`, `skeleton.npz`) and config fingerprinting. |
| `relrel/schema.py` | `RandomSchemaGraphBuilder` (3 table-layout DAGs: `BarabasiAlbert`, `ReverseRandomTree`, `WattsStrogatz`; power-law column counts; sink tables = `Activity`, non-sinks = `Entity`) and `SQLSchemaGraphBuilder` (SQLite/SQLAlchemy DDL parser). |
| `relrel/dag.py` | `DAG_REGISTRY`: `ErdosRenyi`, `BarabasiAlbert`, `RandomTree`, `ReverseRandomTree`, `WattsStrogatz`, `Layered`. The 5 SCM families are all except `WattsStrogatz`. |
| `relrel/bipartite.py` | `sample_bipartite_assignments`: vectorized/indexed hierarchical SBM FK sampler matching PluRel v1.0.0 `get_bipartite_hsbm` without constructing NetworkX bipartite graphs. |
| `relrel/skeleton.py` | `Skeleton` (pre-sampled row counts, unique stable `row_ids`, scalar-or-null `-1` FK arrays, CSR `InverseIndex`, fixed structural metadata) and `check_memory` budget guards. |
| `relrel/rscm.py` | `Column`, `FKRole`, `PathStep`, `EdgeSpec`, `ColumnGraph` (`global-column-dag-v1`), `enumerate_schema_paths` (1- and 2-hop simple paths in either FK direction), and `sample_column_graph` (family-based with locality masking & keyed priority capping, or direct degree-prior / exact `column_edge_budget`). |
| `relrel/relational.py` | `compile_grounding`, `GroundingPlan`, `PreparedPool`, `compile_predicate`, `sample_positions` (Floyd's `O(k)` uniform subset sampler). Implements all 7 grounding patterns (`same_row`, `forward`, `reverse`, `forward_forward`, `forward_reverse_shared_pivot`, `reverse_forward_unique`, `reverse_reverse`), unique-endpoint deduplication, `full` group pooling (float64 accumulation), and `bounded` per-edge parent sampling. |
| `relrel/generation.py` | `GlobalSCMGenerator` (`v1.0.0-global-rscm-v1`), `FixedMLP` (frozen hidden activations), `make_projection`, `make_source`, `calibrate_quantiles`, `CategoricalDecoding` (eager Dirichlet quantile discretization). Supports deterministic replay, optional `fixed_mlp=False`, and `evaluate(interventions=...)`. |
| `relrel/experiments.py` | Direct hyperparameter-driven CLI (`python -m relrel.experiments`) for `synthesize`, `export`, and `generate`. No `SUITE`/`ARM`/`reference` indirection; only `DAG` in `{"local", "global"}`. |
| `relrel/rng.py` | `canonical_json`, `config_fingerprint`, `keyed_seed` (`keyed-sha256-v1`), `scoped_rng` (saves/restores Python, NumPy, PyTorch CPU, and already-initialized CUDA RNGs without initializing CUDA). |
| `relrel/scm.py`, `relrel/ts.py`, `relrel/utils.py` | PluRel v1.0.0 per-table `SCM` (extended with optional eager `categorical_alpha_choices` quantile decoding and `fixed_mlp`), temporal source processes (`TSDataGenerator`, `CategoricalTSDataGenerator`), `MLP`, `CategoricalEncoder`, `CategoricalDecoder`, `TableType`. |
| `scripts/rscm_gen.py` | Standalone CLI for generating `global_rscm` databases to disk (`--tiny`, `--max-hops`, `--parent-policy`, `--schema-file`, etc.). |
| `test/` | Unit and integration tests: `test_rscm.py`, `test_experiment_alignment.py`, `test_dag.py`, `test_dataset.py`, `test_schema.py`, `test_scm.py`, `test_utils.py`. |

## Core Architecture & Invariants

### 1. Three-Stage Separation & Acyclic Grounding
`global_rscm` fixes **schema** -> **FK skeleton** -> **global column DAG** before evaluating any feature cell:
- Every column edge strictly increases global column rank, guaranteeing an acyclic grounded cell graph even with reverse FK traversals (`pk_to_fk`) and table-return chains (e.g., `A.x -> B.y -> A.z`).
- Within-table edges must use the empty path (same-row identity). Cross-table edges use 1- or 2-hop simple paths starting at the target table and ending at the source table (`PathStep.direction` in `{"fk_to_pk", "pk_to_fk"}`).
- Self-referential FKs, same-column cross-row dependencies, and feature-dependent grounding predicates are disallowed.
- Multiple join witnesses reaching the same source row are deduplicated before pooling (`mean` or `sum`). Empty neighborhoods pool to the zero vector while retaining the non-root decoder and explicit noise.

### 2. Determinism, Interventions & Memory Limits
- Never use Python's salted `hash()` for seeds or tie-breaking. Derive all streams via `keyed_seed(seed, ...)` and isolate legacy global-RNG factories inside `scoped_rng(...)`.
- `FixedMLP` draws hidden activations once at construction so repeated `evaluate()` calls and `evaluate(interventions=...)` are deterministic and invariant to `RSCMParams.batch_size`.
- `RSCMParams` enforces explicit memory limits (`index_memory_bytes`, `temp_memory_bytes`, `parent_map_memory_bytes`). **Never silently switch from `full` to `bounded` pooling** when a limit is exceeded—raise `MemoryError`.

### 3. Eager Quantile Decoding
- Controlled by `CategoricalDecoding(alpha_choices)` / `RSCMParams.categorical_alpha_choices` (experiments use `(1.0, 5.0)`).
- When enabled, categorical columns generate continuous scalar scores (both root `TSDataGenerator` and non-root `FixedMLP` decoders) and are discretized via `calibrate_quantiles` (single Dirichlet draw, tie-preserving cutoffs) **before** downstream columns read them.
- On interventions, baseline cutoffs calibrated during observational evaluation are frozen and reused (`GlobalSCMGenerator.category_calibrations`).

### 4. Hyperparameter-Driven Synthesis & Export Contract (`relrel/experiments.py`)
- Generation profile: `GENERATION_PROFILE = "relrel-hyperparam-v1"`.
- **Flat storage layout**: `DATA/synthetic/raw/<NAME>/` and `DATA/synthetic/rt/<NAME>/` (no `SUITE`/`ARM`/`SPLIT` nesting; `NAME` defaults to a slug derived from the hyperparameters).
- **Only two DAG modes**: `DAG=local` (PluRel's `SCM` directly via `SyntheticDataset`) and `DAG=global` (`GlobalSCMGenerator` via `SyntheticDataset`).
- **PluRel v1.0.0 bit-accuracy**: `DAG=local` with `DECODING=original`, `FIXED_MLP=0` (default for `local`), `DATES=activity`, `EXPORT_CAT=plurel_bool` produces databases **bit-for-bit identical** to PluRel v1.0.0 (`2a273cf`), verified directly in `test_experiment_alignment.py` across `seed=0, 1, 10000`.
- **Direct hyperparameters**: `DAG` (`local`|`global`), `DECODING` (`original`|`quantile`), `ALPHA` (`1.0,5.0`), `FIXED_MLP` (`0`|`1`; default `0` for `local`, `1` for `global`), `SAMPLER` (`family`|`direct`), `MAX_HOPS` (`1`|`2`), `COL_PARENT_CAP` (`8`), `DEGREE`, `SCM_FAMILIES`, `PARENT_POLICY` (`bounded`|`full`), `CELL_PARENT_CAP` (`8`), `ROW_POOLING` (`mean`|`sum`), `DATES` (`activity`|`none`), `EXPORT_CAT` (`plurel_bool`|`text`|`raw_int`), `MAX_FKS` (`5`), `COUNT`, `SEED_START`, `WORKERS`, `TINY`, `SCHEMA_FILE`, `MATCH_SEEDS_FROM`, `MATCH_EDGES_FROM`.
- **Rustler compatibility guards**:
  - Tables with `> MAX_FKS` (default `5`) foreign keys are rejected during synthesis (Rustler hardcodes `MAX_F2P_NBRS = 5`).
  - If an undated table has rows where all feature columns are NaN, `export` adds a `_rt_row` string column so `rustler pre` still materializes every physical row node.
- **Export typing (`EXPORT_CAT`)**:
  - `plurel_bool` (default): matches PluRel v1.0.0 `(x > 0).astype(bool)` for all integer categoricals, drops constants (~50/50 `clf`/`reg` targets).
  - `text`: binary `[0, 1]` → `bool` (`clf`); multiclass → string values (text context in RT).
  - `raw_int`: binary `[0, 1]` → `bool`; multiclass kept as `int64` (`SemType::Number` in `rustler pre`).
- **Atomic staging**: `synthesize` and `export` write to temporary directories and atomically rename; existing destinations raise `FileExistsError`.

## Setup, Commands & Verification

### Environment
- Package manager: **Pixi** (`pyproject.toml`, Python >= 3.12). Run `pixi install` from `relrel/`.
- Formatting & linting: **Ruff** (`line-length = 100`, `target-version = "py312"`, rules `E, F, I, UP`; pre-commit configured in `.pre-commit-config.yaml`).

### Standard Make Workflows
```sh
export DATA=/shared/scratch/relrel-rt
make generate NAME=local-orig    DAG=local  DECODING=original COUNT=128 WORKERS=8
make generate NAME=local-quant   DAG=local  DECODING=quantile COUNT=128 WORKERS=8
make generate NAME=global-orig   DAG=global DECODING=original MAX_HOPS=2 COUNT=128 WORKERS=8
make generate NAME=global-quant  DAG=global DECODING=quantile MAX_HOPS=2 COUNT=128 WORKERS=8
make generate NAME=global-direct DAG=global DECODING=quantile SAMPLER=direct MATCH_EDGES_FROM=global-quant WORKERS=8
make generate NAME=holdout-val   DAG=global DECODING=quantile COUNT=16 SEED_START=100000 WORKERS=8
```

### Verification Checklist
1. **Unit, alignment & PluRel v1.0.0 bit-accuracy tests** (`pytest.ini`: `-n 8 --maxfail=1`):
   ```sh
   pixi run pytest
   ```
2. **Lint & format check**:
   ```sh
   pixi run ruff check . && pixi run ruff format --check .
   ```
3. **Tiny generation + export smoke test** (separate `DATA` root):
   ```sh
   make generate DATA=/tmp/relrel-smoke NAME=smoke-local  DAG=local  DECODING=original TINY=1 COUNT=4
   make generate DATA=/tmp/relrel-smoke NAME=smoke-global DAG=global DECODING=quantile TINY=1 COUNT=4
   ```
