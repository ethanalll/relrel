"""Counted, paired synthetic corpora for the RelRel–RT ablations."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table
from torch_frame import stype

from relrel.config import Choices, Config, DatabaseParams
from relrel.dataset import SyntheticDataset
from relrel.generation import CategoricalDecoding, GlobalSCMGenerator
from relrel.local_generation import features, local_frames
from relrel.rng import canonical_json, config_fingerprint, keyed_seed, scoped_rng
from relrel.rscm import Column, sample_column_graph
from relrel.skeleton import Skeleton
from relrel.utils import TableType


@dataclass(frozen=True)
class Arm:
    dag: Literal["local", "global"]
    decoding: CategoricalDecoding
    match_family_edges: bool = False


ORIGINAL = CategoricalDecoding()
QUANTILE = CategoricalDecoding((1.0, 5.0))
GENERATION_PROFILE = "paired-activity-dates-v1"
REFERENCE_REVISIONS = {
    "plurel": "2a273cfd21933ee4893dfcb862a3edaed45ac665",
    "relrel": "40cca23880063cb0d74829ef387e01a028600563",
}

FACTORIAL = {
    "local-original": Arm("local", ORIGINAL),
    "local-quantile": Arm("local", QUANTILE),
    "global-original": Arm("global", ORIGINAL),
    "global-quantile": Arm("global", QUANTILE),
}
SUITES = {
    "factorial": FACTORIAL,
    "sampler": {
        "family": FACTORIAL["global-quantile"],
        "direct": replace(FACTORIAL["global-quantile"], match_family_edges=True),
    },
    "holdout": {"reference": FACTORIAL["local-original"]},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table_controls(dataset, tables, skeleton):
    """Prepare dates and missingness once, before evaluating any arm."""
    timestamps, missing = {}, {}
    for name, info in sorted(tables.items()):
        count = skeleton.row_counts[name]
        if info["type"] == TableType.Activity:
            timestamps[name] = pd.date_range(
                start=dataset.min_timestamp, end=dataset.max_timestamp, periods=count
            )
        missing[name] = {}
        for column, spec in sorted(features(info).items()):
            if spec["_stype"] == stype.numerical:
                with scoped_rng(keyed_seed(dataset.seed, "experiment-missingness", name, column)):
                    share = dataset.config.database_params.column_nan_perc.sample_uniform()
                    missing[name][column] = np.random.choice(
                        count, int(share * count), replace=False
                    )
    return timestamps, missing


def make_case(seed, suite, tiny, output):
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        return generate_case(seed, suite, tiny, Path(tmp), output)


def generate_case(seed, suite, tiny, stage, output):
    config = experiment_config(tiny)
    # Schema, FK links and postprocessing RNG are shared across all arms at this seed.
    dataset = SyntheticDataset(seed, config)
    tables = dataset._prepare_global_schema()
    if any(len(t["fkey_col_to_pkey_table"]) > 5 for t in tables.values()):
        return {"seed": seed, "rejected": "more than five FKs in a table"}
    skeleton = Skeleton.from_schema(tables, config, seed)
    timestamps, missing = table_controls(dataset, tables, skeleton)
    columns = [
        Column(table, col)
        for table, info in sorted(tables.items())
        for col in sorted(features(info))
    ]
    family = sample_column_graph(
        columns, skeleton.roles, config.scm_params, config.dag_params, config.rscm, seed
    )
    arms = SUITES[suite]
    if suite == "holdout":
        dag = "global" if seed % 2 else "local"
        decoder = "quantile" if (seed // 2) % 2 else "original"
        arms = {"reference": FACTORIAL[f"{dag}-{decoder}"]}
    db_id = f"db-{seed:08d}"
    case = {"seed": seed, "id": db_id, "skeleton_sha256": skeleton.fingerprint, "arms": {}}
    for arm, settings in arms.items():
        global_dag = settings.dag == "global"
        direct = settings.match_family_edges
        params = replace(
            config.rscm,
            categorical_alpha_choices=settings.decoding.alpha_choices,
            column_edge_budget=len(family.edges) if direct else None,
        )
        if global_dag:
            graph = family
            if direct:
                graph = sample_column_graph(
                    columns, skeleton.roles, config.scm_params, config.dag_params, params, seed
                )
            if direct and len(graph.edges) != len(family.edges):
                return {
                    "seed": seed,
                    "rejected": "direct edge budget shortfall",
                    "requested_edges": len(family.edges),
                    "realized_edges": len(graph.edges),
                }
            generator = GlobalSCMGenerator(
                tables, graph, skeleton, replace(config, rscm=params), seed
            )
            frames = generator.evaluate()
            causal = generator.sidecar()
            edge_count = len(graph.edges)
        else:
            frames, causal = local_frames(dataset, tables, skeleton, settings.decoding)
            edge_count = None

        database, task_list, categorical_context, constants = {}, [], {}, 0
        for name, info in sorted(tables.items()):
            frame = frames[name].copy()
            pk, fks = info["pkey_col"], info["fkey_col_to_pkey_table"]
            for fk in fks:
                frame[fk] = frame[fk].astype("Int64")
            for col, spec in sorted(features(info).items()):
                values = frame[col]
                if spec["_stype"] == stype.categorical:
                    categories = spec["categories"]
                    if categories == [0, 1]:
                        frame[col] = values.astype(bool)
                        kind = "clf"
                    else:
                        frame[col] = values.map(lambda index: categories[int(index)])
                        kind = None
                        categorical_context.setdefault(name, []).append(col)
                    constants += int(values.nunique() < 2)
                else:
                    frame[col] = values.astype(np.float64)
                    frame.loc[missing[name][col], col] = np.nan
                    kind = "reg"
                if kind and frame[col].nunique(dropna=True) > 1:
                    task_list.append([db_id, name, col, [], kind])
            # Ensure PK-only rows still get physical identities in the paper preprocessor.
            if not frame[list(features(info))].notna().any(axis=1).all():
                frame["_rt_row"] = f"row of {name}"
            time_col = None
            if name in timestamps:
                time_col = "date"
                frame[time_col] = timestamps[name]
            ordered = [pk, *sorted(fks), *sorted(features(info))]
            ordered.extend(col for col in frame if col not in ordered)
            database[name] = Table(
                df=frame[ordered], pkey_col=pk, fkey_col_to_pkey_table=fks, time_col=time_col
            )
        if not task_list:
            return {"seed": seed, "rejected": f"no varying numeric or Boolean targets in {arm}"}
        dest = stage / arm / db_id
        dest.mkdir(parents=True)
        Database(database).save(dest / "db")
        write_json(dest / "causal.json", causal)
        case["arms"][arm] = {
            "global_dag": global_dag,
            "quantiles": settings.decoding.alpha_choices is not None,
            "sampler": ("direct" if direct else "family") if global_dag else None,
            "edges": edge_count,
            "requested_edges": len(family.edges) if direct else None,
            "constant_categorical_columns": constants,
            "categorical_context": categorical_context,
            "tasks": task_list,
        }
    common_targets = set.intersection(
        *({tuple(task[:3]) for task in arm_info["tasks"]} for arm_info in case["arms"].values())
    )
    if not common_targets:
        return {"seed": seed, "rejected": "no shared varying numeric or Boolean targets"}
    for arm, arm_info in case["arms"].items():
        arm_info["tasks"] = [t for t in arm_info["tasks"] if tuple(t[:3]) in common_targets]
        (output / arm).mkdir(exist_ok=True)
        (stage / arm / db_id).rename(output / arm / db_id)
    return case


def experiment_config(tiny):
    prior = DatabaseParams()
    if tiny:
        prior = replace(
            prior,
            num_tables_choices=Choices("set", [3]),
            num_cols_choices=Choices("range", [4, 4]),
            num_rows_entity_table_choices=Choices("set", [12]),
            num_rows_activity_table_choices=Choices("set", [24]),
        )
    return Config(database_params=prior, generation_mode="global_rscm")


class RelRelReference(SyntheticDataset):
    """Use the standalone RelRel revision's categorical export policy."""

    def generation_identity(self):
        identity = super().generation_identity()
        identity.pop("fingerprint")
        identity["categorical_export_version"] = 2
        return {**identity, "fingerprint": config_fingerprint(identity)}

    def process_categorical_data(self, frame, feature_columns, pkey_col, fkey_cols):
        for column, info in feature_columns.items():
            if info["_stype"] != stype.categorical:
                continue
            categories = info["categories"]
            if categories == [0, 1]:
                frame[column] = frame[column].astype(bool)
            elif isinstance(categories[0], str):
                frame[column] = frame[column].map(lambda index: categories[index])
        return frame


def generate_reference(root, reference, count, seed, tiny):
    """Run the standalone generator with its own corpus policy."""
    output = root / "synthetic/reference" / reference
    if output.exists():
        raise FileExistsError(f"{output} already exists. Use a fresh DATA path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = experiment_config(tiny)
    if reference == "plurel":
        config = replace(config, generation_mode="legacy_scm")
        dataset_cls = SyntheticDataset
    else:
        config = replace(
            config, rscm=replace(config.rscm, categorical_alpha_choices=QUANTILE.alpha_choices)
        )
        dataset_cls = RelRelReference
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        databases = []
        for offset in range(seed, seed + count):
            name = f"db-{offset:08d}"
            cache = stage / name
            dataset = dataset_cls(offset, replace(config, cache_dir=str(cache)))
            dataset.get_db(upto_test_timestamp=False)
            databases.append({"id": name, "seed": offset})
        write_json(
            stage / "corpus.json",
            {
                "version": 1,
                "reference": reference,
                "reference_revision": REFERENCE_REVISIONS[reference],
                "config": json.loads(canonical_json(config)),
                "databases": databases,
            },
        )
        stage.rename(output)
    print(output)


def configure_worker():
    torch.set_num_threads(1)


def synthesize(root, suite, split, count, seed, workers, tiny):
    out = root / "synthetic/raw" / suite / split
    if out.exists():
        raise FileExistsError(f"{out} already exists. Use a fresh DATA path for a new corpus.")
    out.parent.mkdir(parents=True, exist_ok=True)
    config = experiment_config(tiny)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    source = hashlib.sha256()
    for path in sorted((repo / "relrel").glob("*.py")):
        source.update(path.read_bytes())

    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        accepted, rejected = [], []
        # Process one batch at a time and accept results in seed order.
        offset, limit = seed, seed + count * 10
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"), initializer=configure_worker
        ) as pool:
            while len(accepted) < count and offset < limit:
                stop = min(offset + count - len(accepted), limit)
                cases = pool.map(
                    make_case,
                    range(offset, stop),
                    repeat(suite),
                    repeat(tiny),
                    repeat(stage),
                )
                for case in cases:
                    if "rejected" in case:
                        rejected.append(case)
                    else:
                        accepted.append(case)
                    print(f"{case['seed']} {case.get('rejected', 'accepted')}", flush=True)
                offset = stop
        if len(accepted) < count:
            raise ValueError(
                f"Only {len(accepted)} paired databases accepted out of {limit - seed}"
            )
        manifest = {
            "version": 1,
            "suite": suite,
            "split": split,
            "count": count,
            "seed_start": seed,
            "generation_profile": GENERATION_PROFILE,
            "revision": revision,
            "source_sha256": source.hexdigest(),
            "config": json.loads(canonical_json(config)),
            "cases": accepted,
            "rejected": rejected,
        }
        write_json(stage / "corpus.json", manifest)
        stage.rename(out)
    print(out)


def export(root, suite, split):
    source = root / "synthetic/raw" / suite / split
    output = root / "synthetic/rt" / suite / split
    manifest_bytes = (source / "corpus.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "data"
        for arm in SUITES[suite]:
            entries, tasks = [], []
            for case in manifest["cases"]:
                db = case["id"]
                dest = stage / arm / db
                dest.mkdir(parents=True)
                raw_db = Database.load(source / arm / db / "db")
                info = case["arms"][arm].copy()
                for table_name, columns in info.pop("categorical_context").items():
                    table = raw_db.table_dict[table_name]
                    for column in columns:
                        table.df[column] = table.df[column].map(str)
                raw_db.save(dest / "db")
                tasks.extend(info.pop("tasks"))
                entries.append({"id": db, "seed": case["seed"], **info})
            write_json(
                stage / arm / "corpus.json",
                {
                    "version": 1,
                    "suite": suite,
                    "split": split,
                    "arm": arm,
                    "generation_profile": manifest.get("generation_profile", "legacy-static"),
                    "generation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "databases": entries,
                    "tasks": tasks,
                },
            )
        stage.rename(output)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthesize", "export", "reference"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--reference", choices=REFERENCE_REVISIONS)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if args.command == "reference":
        if args.reference is None:
            parser.error("Choose --reference for standalone generation")
        count = args.count if args.count is not None else 1
        seed = args.seed_start if args.seed_start is not None else 0
        if count < 1 or seed < 0 or seed + count > 2**32:
            parser.error("Invalid reference count or seed range")
        generate_reference(root, args.reference, count, seed, args.tiny)
        return
    if args.suite is None or args.split is None:
        parser.error("Choose --suite and --split for the paired corpus")
    if (args.suite == "holdout") != (args.split == "val"):
        parser.error("Use holdout with val and factorial/sampler with train")
    if args.command == "synthesize":
        count = args.count if args.count is not None else (16 if args.split == "val" else 128)
        seed = (
            args.seed_start
            if args.seed_start is not None
            else (100_000 if args.split == "val" else 10_000)
        )
        if count < 1 or args.workers < 1 or not 0 <= seed <= 2**32 - count * 10:
            parser.error("Invalid count, workers or seed range")
        synthesize(root, args.suite, args.split, count, seed, args.workers, args.tiny)
    else:
        export(root, args.suite, args.split)


if __name__ == "__main__":
    main()
