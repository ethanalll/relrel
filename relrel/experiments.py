"""Hyperparameter-driven synthetic corpus synthesis and RT export for RelRel."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from relbench.base import Database

from relrel.config import Choices, Config, DatabaseParams, RSCMParams, SCMParams
from relrel.dataset import SyntheticDataset
from relrel.rng import canonical_json

GENERATION_PROFILE = "relrel-hyperparam-v1"
SCM_FAMILIES = (
    "ErdosRenyi",
    "BarabasiAlbert",
    "RandomTree",
    "ReverseRandomTree",
    "Layered",
)


@dataclass(frozen=True)
class RunSpec:
    dag: Literal["local", "global"] = "global"
    decoding: Literal["original", "quantile"] = "quantile"
    alpha: tuple[float, ...] = (1.0, 5.0)
    fixed_mlp: bool | None = None
    sampler: Literal["family", "direct"] = "family"
    max_hops: int = 2
    col_parent_cap: int = 8
    degree: tuple[float, ...] | None = None
    scm_families: tuple[str, ...] | None = None
    parent_policy: Literal["bounded", "full"] = "bounded"
    cell_parent_cap: int = 8
    row_pooling: Literal["mean", "sum"] = "mean"
    dates: Literal["activity", "none"] = "activity"
    export_cat: Literal["plurel_bool", "text", "raw_int"] = "plurel_bool"
    max_fks: int = 5
    tiny: bool = False
    schema_file: str | None = None

    @property
    def resolved_fixed_mlp(self) -> bool:
        if self.fixed_mlp is not None:
            return self.fixed_mlp
        return self.dag == "global"

    def validate(self, *, match_edges: bool = False) -> None:
        if self.dag not in ("local", "global"):
            raise ValueError("dag must be 'local' or 'global'")
        if self.decoding not in ("original", "quantile"):
            raise ValueError("decoding must be 'original' or 'quantile'")
        if self.dates not in ("activity", "none"):
            raise ValueError("dates must be 'activity' or 'none'")
        if self.export_cat not in ("plurel_bool", "text", "raw_int"):
            raise ValueError("export_cat must be 'plurel_bool', 'text', or 'raw_int'")
        if self.max_fks < 0:
            raise ValueError("max_fks must be >= 0")
        if self.scm_families is not None:
            unknown = set(self.scm_families) - set(SCM_FAMILIES)
            if not self.scm_families or unknown:
                raise ValueError(f"Unknown SCM families: {sorted(unknown)}")
        if self.dag == "local":
            if self.sampler != "family" or self.degree is not None or match_edges:
                raise ValueError(
                    "--sampler=direct, --degree, and --match-edges-from require --dag=global"
                )
        else:
            if self.sampler == "family" and (self.degree is not None or match_edges):
                raise ValueError("--degree and --match-edges-from require --sampler=direct")
            if self.sampler == "direct" and self.degree is None and not match_edges:
                raise ValueError("--sampler=direct requires --degree or --match-edges-from")
            if self.degree is not None and match_edges:
                raise ValueError("Provide either --degree or --match-edges-from, not both")

    def default_name(self) -> str:
        parts = [self.dag, self.decoding]
        if self.dag == "global":
            if self.sampler != "family":
                parts.append(self.sampler)
            if self.max_hops != 2:
                parts.append(f"h{self.max_hops}")
            if self.parent_policy != "bounded":
                parts.append(self.parent_policy)
        if self.fixed_mlp is not None:
            parts.append("fixedmlp" if self.fixed_mlp else "randmlp")
        if self.dates != "activity":
            parts.append("nodates")
        if self.export_cat != "plurel_bool":
            parts.append(self.export_cat)
        if self.tiny:
            parts.append("tiny")
        return "-".join(parts)

    def to_config(self, *, edge_budget: int | None = None) -> Config:
        db_params = DatabaseParams()
        if self.tiny:
            db_params = replace(
                db_params,
                num_tables_choices=Choices("set", [3]),
                num_cols_choices=Choices("range", [4, 4]),
                num_rows_entity_table_choices=Choices("set", [12]),
                num_rows_activity_table_choices=Choices("set", [24]),
            )
        scm_params = SCMParams()
        if self.scm_families is not None:
            scm_params = replace(
                scm_params,
                scm_layout_choices=Choices("set", list(self.scm_families)),
            )
        rscm = RSCMParams(
            max_hops=self.max_hops,
            column_parent_cap=self.col_parent_cap,
            parent_policy=self.parent_policy,
            cell_parent_cap_per_edge=self.cell_parent_cap,
            row_pooling=self.row_pooling,
            categorical_alpha_choices=self.alpha if self.decoding == "quantile" else None,
            column_degree_choices=self.degree
            if (self.dag == "global" and edge_budget is None)
            else None,
            column_edge_budget=edge_budget if self.dag == "global" else None,
        )
        return Config(
            database_params=db_params,
            scm_params=scm_params,
            schema_file=self.schema_file,
            generation_mode="global_rscm" if self.dag == "global" else "legacy_scm",
            rscm=rscm,
            fixed_mlp=self.resolved_fixed_mlp,
            include_activity_dates=self.dates == "activity",
            categorical_export=self.export_cat,
        )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_floats(raw: str | None) -> tuple[float, ...] | None:
    if not raw:
        return None
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_strings(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise ValueError(f"Expected 0/1 or true/false, got {raw!r}")


def resolve_manifest(root: Path, ref: str) -> dict:
    candidate = Path(ref).expanduser()
    if candidate.is_file():
        path = candidate
    elif (candidate / "corpus.json").is_file():
        path = candidate / "corpus.json"
    else:
        path = root / "synthetic/raw" / ref / "corpus.json"
    if not path.is_file():
        raise FileNotFoundError(f"Could not locate reference corpus manifest for {ref!r} at {path}")
    return json.loads(path.read_text())


def extract_tasks(db: Database, db_id: str) -> tuple[list[list], int, int]:
    """Identify eligible varying Boolean (clf) and floating (reg) RT targets."""
    tasks: list[list] = []
    clf_count = 0
    reg_count = 0
    for table_name, table in sorted(db.table_dict.items()):
        excluded = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
        for col in sorted(table.df.columns):
            if col in excluded or col == "_rt_row":
                continue
            series = table.df[col]
            if pd.api.types.is_bool_dtype(series):
                kind = "clf"
            elif pd.api.types.is_float_dtype(series):
                kind = "reg"
            else:
                continue
            if series.nunique(dropna=True) > 1:
                tasks.append([db_id, table_name, col, [], kind])
                if kind == "clf":
                    clf_count += 1
                else:
                    reg_count += 1
    return tasks, clf_count, reg_count


def generate_one(seed: int, spec: RunSpec, edge_budget: int | None, output: Path) -> dict:
    config = spec.to_config(edge_budget=edge_budget)
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    if spec.max_fks > 0 and any(
        len(table.fkey_col_to_pkey_table) > spec.max_fks for table in db.table_dict.values()
    ):
        return {"seed": seed, "rejected": f"more than {spec.max_fks} FKs in a table"}
    if edge_budget is not None:
        realized = len(dataset.column_graph.edges)
        if realized != edge_budget:
            return {
                "seed": seed,
                "rejected": "direct edge budget shortfall",
                "requested_edges": edge_budget,
                "realized_edges": realized,
            }

    db_id = f"db-{seed:08d}"
    tasks, clf_count, reg_count = extract_tasks(db, db_id)
    if not tasks:
        return {"seed": seed, "rejected": "no varying numeric or Boolean targets"}

    with tempfile.TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp) / db_id
        stage.mkdir(parents=True)
        db.save(stage / "db")
        if spec.dag == "global":
            causal = dataset.causal_metadata
            edges = len(dataset.column_graph.edges)
            skeleton_sha = dataset.skeleton.fingerprint
        else:
            causal = {"local_scm": dataset.local_diagnostics}
            edges = None
            skeleton_sha = None
        write_json(stage / "causal.json", causal)
        stage.rename(output / db_id)

    return {
        "id": db_id,
        "seed": seed,
        "dag": spec.dag,
        "decoding": spec.decoding,
        "sampler": spec.sampler if spec.dag == "global" else None,
        "edges": edges,
        "requested_edges": edge_budget,
        "skeleton_sha256": skeleton_sha,
        "clf_tasks": clf_count,
        "reg_tasks": reg_count,
        "tasks": tasks,
    }


def _worker_call(args):
    seed, spec, edge_budget, output = args
    return generate_one(seed, spec, edge_budget, output)


def configure_worker() -> None:
    torch.set_num_threads(1)


def synthesize(
    root: Path,
    name: str,
    spec: RunSpec,
    *,
    count: int,
    seed_start: int,
    workers: int,
    match_seeds_from: str | None = None,
    match_edges_from: str | None = None,
) -> Path:
    if Path(name).name != name or name in (".", "..", ""):
        raise ValueError("NAME must be a single directory name")
    spec.validate(match_edges=match_edges_from is not None)
    out = root / "synthetic/raw" / name
    if out.exists():
        raise FileExistsError(f"{out} already exists. Choose a new NAME or DATA root.")
    out.parent.mkdir(parents=True, exist_ok=True)

    fixed_seeds: list[tuple[int, int | None]] | None = None
    if match_edges_from is not None:
        ref = resolve_manifest(root, match_edges_from)
        fixed_seeds = []
        for entry in ref["databases"]:
            if entry.get("edges") is None:
                raise ValueError(
                    f"Reference corpus {match_edges_from!r} has no global edge counts to match"
                )
            fixed_seeds.append((int(entry["seed"]), int(entry["edges"])))
    elif match_seeds_from is not None:
        ref = resolve_manifest(root, match_seeds_from)
        fixed_seeds = [(int(entry["seed"]), None) for entry in ref["databases"]]
    if fixed_seeds is not None and not fixed_seeds:
        raise ValueError("Reference manifest contains no databases to match")

    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    source = hashlib.sha256()
    for path in sorted((repo / "relrel").glob("*.py")):
        source.update(path.read_bytes())

    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        accepted, rejected = [], []
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"), initializer=configure_worker
        ) as pool:
            if fixed_seeds is not None:
                jobs = [(s, spec, b, stage) for s, b in fixed_seeds]
                for entry in pool.map(_worker_call, jobs):
                    if "rejected" in entry:
                        rejected.append(entry)
                    else:
                        accepted.append(entry)
                    print(f"{entry['seed']} {entry.get('rejected', 'accepted')}", flush=True)
                if len(accepted) != len(fixed_seeds):
                    raise ValueError(
                        f"Matched run accepted {len(accepted)}/{len(fixed_seeds)} seeds; "
                        f"rejected={rejected[:5]}"
                    )
            else:
                offset, limit = seed_start, seed_start + count * 10
                while len(accepted) < count and offset < limit:
                    stop = min(offset + count - len(accepted), limit)
                    jobs = [(s, spec, None, stage) for s in range(offset, stop)]
                    for entry in pool.map(_worker_call, jobs):
                        if "rejected" in entry:
                            rejected.append(entry)
                        else:
                            accepted.append(entry)
                        print(f"{entry['seed']} {entry.get('rejected', 'accepted')}", flush=True)
                    offset = stop
                if len(accepted) < count:
                    raise ValueError(
                        f"Only {len(accepted)} databases accepted out of {limit - seed_start}"
                    )

        rejection_counts: dict[str, int] = {}
        for item in rejected:
            reason = item["rejected"]
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        manifest = {
            "version": 2,
            "name": name,
            "generation_profile": GENERATION_PROFILE,
            "count": len(accepted),
            "seed_start": seed_start if fixed_seeds is None else accepted[0]["seed"],
            "match_seeds_from": match_seeds_from,
            "match_edges_from": match_edges_from,
            "revision": revision,
            "source_sha256": source.hexdigest(),
            "hyperparams": {**asdict(spec), "resolved_fixed_mlp": spec.resolved_fixed_mlp},
            "config": json.loads(canonical_json(spec.to_config())),
            "rejection_counts": rejection_counts,
            "databases": accepted,
            "rejected": rejected,
        }
        write_json(stage / "corpus.json", manifest)
        stage.rename(out)
    print(out)
    return out


def export(root: Path, name: str) -> Path:
    if Path(name).name != name or name in (".", "..", ""):
        raise ValueError("NAME must be a single directory name")
    source = root / "synthetic/raw" / name
    output = root / "synthetic/rt" / name
    manifest_bytes = (source / "corpus.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} already exists")

    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        entries, tasks = [], []
        for db_entry in manifest["databases"]:
            db_id = db_entry["id"]
            dest = stage / db_id
            dest.mkdir(parents=True)
            raw_db = Database.load(source / db_id / "db")
            for table_name, table in raw_db.table_dict.items():
                for fk in table.fkey_col_to_pkey_table:
                    if not pd.api.types.is_integer_dtype(table.df[fk]):
                        table.df[fk] = table.df[fk].astype("Int64")
                excluded = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
                feature_cols = [c for c in table.df.columns if c not in excluded]
                # Ensure rows whose features are all NaN still materialize in rustler pre.
                if (not feature_cols and table.time_col is None) or (
                    feature_cols
                    and table.time_col is None
                    and not table.df[feature_cols].notna().any(axis=1).all()
                ):
                    table.df["_rt_row"] = f"row of {table_name}"
            raw_db.save(dest / "db")
            entry_copy = dict(db_entry)
            tasks.extend(entry_copy.pop("tasks"))
            entries.append(entry_copy)

        write_json(
            stage / "corpus.json",
            {
                "version": 2,
                "name": name,
                "generation_profile": manifest.get("generation_profile", GENERATION_PROFILE),
                "generation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "hyperparams": manifest.get("hyperparams", {}),
                "databases": entries,
                "tasks": tasks,
            },
        )
        stage.rename(output)
    print(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthesize", "export", "generate"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--name", default="")
    parser.add_argument("--dag", choices=("local", "global"), default="global")
    parser.add_argument("--decoding", choices=("original", "quantile"), default="quantile")
    parser.add_argument("--alpha", default="1.0,5.0")
    parser.add_argument("--fixed-mlp", default="")
    parser.add_argument("--sampler", choices=("family", "direct"), default="family")
    parser.add_argument("--max-hops", type=int, choices=(1, 2), default=2)
    parser.add_argument("--col-parent-cap", type=int, default=8)
    parser.add_argument("--degree", default="")
    parser.add_argument("--scm-families", default="")
    parser.add_argument("--parent-policy", choices=("bounded", "full"), default="bounded")
    parser.add_argument("--cell-parent-cap", type=int, default=8)
    parser.add_argument("--row-pooling", choices=("mean", "sum"), default="mean")
    parser.add_argument("--dates", choices=("activity", "none"), default="activity")
    parser.add_argument(
        "--export-cat", choices=("plurel_bool", "text", "raw_int"), default="plurel_bool"
    )
    parser.add_argument("--max-fks", type=int, default=5)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--schema-file", default="")
    parser.add_argument("--match-seeds-from", default="")
    parser.add_argument("--match-edges-from", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    if args.command == "export":
        if not args.name:
            parser.error("--name is required for export")
        export(root, args.name)
        return

    alpha = parse_floats(args.alpha) if args.decoding == "quantile" else (1.0, 5.0)
    if args.decoding == "quantile" and not alpha:
        parser.error("--alpha must provide at least one positive float for quantile decoding")
    spec = RunSpec(
        dag=args.dag,
        decoding=args.decoding,
        alpha=alpha,
        fixed_mlp=parse_optional_bool(args.fixed_mlp),
        sampler=args.sampler,
        max_hops=args.max_hops,
        col_parent_cap=args.col_parent_cap,
        degree=parse_floats(args.degree),
        scm_families=parse_strings(args.scm_families),
        parent_policy=args.parent_policy,
        cell_parent_cap=args.cell_parent_cap,
        row_pooling=args.row_pooling,
        dates=args.dates,
        export_cat=args.export_cat,
        max_fks=args.max_fks,
        tiny=args.tiny,
        schema_file=args.schema_file or None,
    )
    name = args.name or spec.default_name()
    if args.count < 1 or args.workers < 1 or not 0 <= args.seed_start <= 2**32 - args.count * 10:
        parser.error("Invalid count, workers, or seed range")
    try:
        spec.validate(match_edges=bool(args.match_edges_from))
    except ValueError as exc:
        parser.error(str(exc))

    synthesize(
        root,
        name,
        spec,
        count=args.count,
        seed_start=args.seed_start,
        workers=args.workers,
        match_seeds_from=args.match_seeds_from or None,
        match_edges_from=args.match_edges_from or None,
    )
    if args.command == "generate":
        export(root, name)


if __name__ == "__main__":
    main()
