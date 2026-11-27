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
| `relrel/generation.py` | `GlobalSCMGenerator` (`v1.0.0-global-rscm-v1`), `FixedMLP` (frozen hidden activations), `make_projection`, `make_source`, `calibrate_quantiles`, `CategoricalDecoding` (eager Dirichlet quantile discretization). Supports deterministic replay and `evaluate(interventions=...)`. |
| `relrel/local_generation.py` | `local_frames`: per-table SCM evaluation for matched ablation arms. Preserves PluRel latent nodes and dense forward-FK collation embeddings while sharing the pre-sampled `Skeleton`, `FixedMLP`, and optional eager quantile decoding. |
| `relrel/experiments.py` | CLI (`python -m relrel.experiments`) for paired corpus synthesis (`synthesize`), RT Parquet/manifest export (`export`), and standalone historical reference corpora (`reference` for `plurel` and `relrel`). |
| `relrel/rng.py` | `canonical_json`, `config_fingerprint`, `keyed_seed` (`keyed-sha256-v1`), `scoped_rng` (saves/restores Python, NumPy, PyTorch CPU, and already-initialized CUDA RNGs without initializing CUDA). |
| `relrel/scm.py`, `relrel/ts.py`, `relrel/utils.py` | Legacy PluRel v1.0.0 per-table `SCM`, temporal source processes (`TSDataGenerator`, `CategoricalTSDataGenerator`), `MLP`, `CategoricalEncoder`, `CategoricalDecoder`, `TableType`. |
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

### 4. Paired Ablation Suites & Export Contract (`relrel/experiments.py`)
- Generation profile: `GENERATION_PROFILE = "paired-activity-dates-v1"`.
- **Suites**:
  - `factorial` (`split=train`, default `COUNT=128`, `SEED_START=10000`): 4 arms — `local-original`, `local-quantile`, `global-original`, `global-quantile`.
  - `sampler` (`split=train`, default `COUNT=128`, `SEED_START=10000`): 2 arms — `family` (global-quantile with 5-family proposal + locality mask) vs. `direct` (global-quantile with `column_edge_budget = len(family.edges)`). Seeds where the direct sampler cannot meet the exact edge budget are recorded in `rejected` and replaced in seed order.
  - `holdout` (`split=val`, default `COUNT=16`, `SEED_START=100000`): single `reference` arm cycling across the 4 factorial settings (`global` = `bool(seed % 2)`, `quantile` = `bool((seed // 2) % 2)`). Keep train and holdout seed ranges disjoint.
- **Paired controls per accepted seed**: identical schema, FK skeleton, feature inventory, candidate Boolean/numeric columns, activity-table timestamps (`date` via `pd.date_range`), numeric missingness masks (`experiment-missingness`), and frozen MLP activations. Each arm retains all of its own varying Boolean (`clf`) and numerical (`reg`) targets rather than intersecting across arms (so post-treatment binary collapse in `original` arms never censors `quantile` arms); manifests record per-arm `clf_tasks`, `reg_tasks`, `constant_binary_columns`, `constant_categorical_columns`, case-level `shared_targets`, and `rejection_counts`.
- **PluRel v1.0.0 bit-accuracy**: `legacy_scm` (`SyntheticDataset(seed, Config(generation_mode="legacy_scm"))` and `make reference REFERENCE=plurel`) preserves exact bit-for-bit parity with PluRel v1.0.0 (`2a273cf`), including `seed=0`. In `global_rscm`, `_prepare_global_schema` and `sample_column_graph` pass `seed=None` into `RandomSchemaGraphBuilder`/`DAG` inside `scoped_rng(...)` so keyed RNG streams are never clobbered, and `ErdosRenyi` advances its retry seed only after 64 consecutive failed attempts to prevent fixed-seed deadlocks.
- **Rustler compatibility guards**:
  - Tables with `> 5` foreign keys are rejected during synthesis (Rustler hardcodes `MAX_F2P_NBRS = 5`).
  - If any table has rows where all feature columns are NaN, export adds a `_rt_row` string column so `rustler pre` still materializes every physical row node.
- **Export typing policy**:
  - Binary categoricals (`categories == [0, 1]`) -> `bool` (`clf` target).
  - Multiclass categoricals (`len(categories) > 2` or string categories) -> string values (`categorical_context`, text context in RT, excluded from targets).
  - Numericals -> `float64` with paired NaN missingness (`reg` target).
- **Atomic staging**: `synthesize`, `export`, and `reference` write to a temporary directory and atomically rename; existing destinations raise `FileExistsError`. Always use a fresh `DATA` root for a new corpus.
- **Historical standalone references** (`make reference`):
  - `REFERENCE=plurel`: `legacy_scm`, dated activity tables, per-call MLP activations (`categorical_export_policy: "plurel_v1_gt0_bool"`, matches PluRel v1.0.0 `2a273cf` bit-for-bit).
  - `REFERENCE=relrel`: `RelRelReference` (`categorical_export_version=2`, `categorical_export_policy: "binary_bool_multiclass_int64"`), `global_rscm` with eager quantiles, static timestamp-free tables (`time_col=None`), retains multiclass integer columns (matches RelRel `40cca23`; note `rustler pre` treats raw `Int64` features as `SemType::Number`).

## Setup, Commands & Verification

### Environment
- Package manager: **Pixi** (`pyproject.toml`, Python >= 3.12). Run `pixi install` from `relrel/`.
- Formatting & linting: **Ruff** (`line-length = 100`, `target-version = "py312"`, rules `E, F, I, UP`; pre-commit configured in `.pre-commit-config.yaml`).

### Standard Make Workflows
```sh
export DATA=/shared/scratch/relrel-rt
make synthesize SUITE=factorial COUNT=128 WORKERS=8
make export SUITE=factorial

make synthesize SUITE=sampler COUNT=128 WORKERS=8
make export SUITE=sampler

make synthesize SUITE=holdout SPLIT=val COUNT=16 WORKERS=8
make export SUITE=holdout SPLIT=val

# Optional standalone references
make reference REFERENCE=plurel COUNT=4 SEED_START=10000
make reference REFERENCE=relrel COUNT=4 SEED_START=10000
```

### Verification Checklist
1. **Unit & alignment tests** (parallelized via `pytest.ini` with `-n 8 --maxfail=1`):
   ```sh
   pixi run pytest
   ```
2. **Lint & format check**:
   ```sh
   pixi run ruff check . && pixi run ruff format --check .
   ```
3. **Tiny synthesis + export smoke test** (always use a separate `DATA` root):
   ```sh
   make synthesize DATA=/tmp/relrel-smoke SUITE=factorial TINY=1 COUNT=4
   make export DATA=/tmp/relrel-smoke SUITE=factorial
   ```
