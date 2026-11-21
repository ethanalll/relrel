"""Counted, paired synthetic corpora for the RelRel–RT ablations."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table
from torch_frame import stype

from relrel.config import Choices, Config, DatabaseParams
from relrel.dataset import SyntheticDataset
from relrel.generation import (
    FixedMLP,
    GlobalSCMGenerator,
    calibrate_quantiles,
    make_projection,
    make_source,
)
from relrel.rng import canonical_json, keyed_seed, scoped_rng
from relrel.rscm import Column, sample_column_graph
from relrel.scm import SCM
from relrel.skeleton import Skeleton
from relrel.utils import MLP

FACTORIAL = {
    "local-original": (False, False, False),
    "local-quantile": (False, True, False),
    "global-original": (True, False, False),
    "global-quantile": (True, True, False),
}
SUITES = {
    "factorial": FACTORIAL,
    "sampler": {"family": (True, True, False), "direct": (True, True, True)},
    "holdout": {"reference": (None, None, False)},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def features(info):
    return {
        name: spec
        for name, spec in info["columns"].items()
        if name != info["pkey_col"] and name not in info["fkey_col_to_pkey_table"]
    }


def fixed_projection(projection, params):
    if isinstance(projection, MLP):
        return FixedMLP(projection, params)
    projection.mlp = FixedMLP(projection.mlp, params)
    return projection


def local_frames(dataset, tables, skeleton, alpha_choices):
    """Evaluate per-table SCMs eagerly, with their original dense FK inputs."""
    seed, params = dataset.seed, dataset.config.scm_params
    scms, frames, diagnostics = {}, {}, {}
    for table_id in nx.topological_sort(dataset.table_relationships):
        info = dataset.table_relationships.nodes[table_id]
        name = info["name"]
        columns = features(info)
        fk_tables = info["fkey_col_to_pkey_table"]
        children = [
            dataset.table_relationships.nodes[child]["name"]
            for child in sorted(dataset.table_relationships.successors(table_id))
        ]
        n = skeleton.row_counts[name]
        with scoped_rng(keyed_seed(seed, "local-scm", name)):
            scm = SCM(
                name,
                children,
                columns,
                info["pkey_col"],
                fk_tables,
                {parent: scms[parent] for parent in fk_tables.values()},
                params,
                dataset.config.dag_params,
            )
        graph = scm.dag.graph
        with scoped_rng(keyed_seed(seed, "local-activations", name)):
            for _, attrs in graph.nodes(data=True):
                attrs["decoder"] = fixed_projection(attrs["decoder"], params)
                for key, encoder in attrs.get("collation_encoders", {}).items():
                    attrs["collation_encoders"][key] = fixed_projection(encoder, params)
            for _, _, attrs in graph.edges(data=True):
                attrs["encoder"] = fixed_projection(attrs["encoder"], params)

        frame = pd.DataFrame({info["pkey_col"]: np.arange(n, dtype=np.int64)})
        foreign_input = torch.zeros((n, params.mlp_emb_dim))
        for role in skeleton.roles:
            if role.child_table != name:
                continue
            parent = role.parent_table
            offsets = skeleton.forward(role)
            frame[role.name] = pd.array(np.where(offsets >= 0, offsets, None), dtype="Int64")
            present = offsets >= 0
            parent_scm = scms[parent]
            parent_cols = features(tables[parent])
            for node in parent_scm.col_nodes:
                attrs = parent_scm.dag.graph.nodes[node]
                column = attrs["col_name"]
                encoder = attrs["collation_encoders"][(parent, name)]
                values = torch.from_numpy(frames[parent][column].to_numpy(copy=True))
                if parent_cols[column]["_stype"] == stype.numerical:
                    values = values.float().reshape(-1, 1)
                foreign_input[present] += encoder(values[offsets[present]]) / len(parent_cols)

        values, calibrations = {}, {}
        for node in nx.topological_sort(graph):
            attrs = graph.nodes[node]
            numerical = attrs["_stype"] == stype.numerical
            quantile = not numerical and alpha_choices is not None
            kind = {"_stype": attrs["_stype"], "categories": None}
            if not numerical:
                kind["categories"] = range(attrs["num_categories"])
            with scoped_rng(keyed_seed(seed, "local-mechanism", name, int(node))):
                if graph.in_degree(node) == 0:
                    source = make_source(params, kind, n, info["type"], scalar_output=quantile)
                    value = torch.tensor(
                        [source.get_value(row) for row in range(n)],
                        dtype=torch.float32 if numerical or quantile else torch.long,
                    )
                else:
                    noise = attrs["noise_dist"].sample((n, params.mlp_emb_dim)).reshape(n, -1)
                    latent = foreign_input + noise / params.mlp_emb_dim
                    for parent in graph.predecessors(node):
                        edge = graph.edges[parent, node]
                        latent += edge["weight"] * edge["encoder"](values[parent])
                    decoder = (
                        make_projection(params, kind, decoder=True, scalar_output=True)
                        if quantile
                        else attrs["decoder"]
                    )
                    value = decoder(latent).reshape(-1)
            if quantile:
                rng = np.random.default_rng(keyed_seed(seed, "local-quantiles", name, int(node)))
                value, calibration = calibrate_quantiles(
                    value, attrs["num_categories"], alpha_choices, rng
                )
                calibrations[attrs.get("col_name", f"latent:{node}")] = calibration
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite feature in {name}, node {node}")
            values[node] = value.reshape(n, 1) if numerical else value
            if node in scm.col_nodes:
                frame[attrs["col_name"]] = value.numpy()

        scm.df = frame
        scms[name], frames[name] = scm, frame
        diagnostics[name] = {"local_edges": graph.number_of_edges(), "quantiles": calibrations}
    return frames, {"local_scm": diagnostics}


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
    columns = [
        Column(table, col)
        for table, info in sorted(tables.items())
        for col in sorted(features(info))
    ]
    family = sample_column_graph(
        columns, skeleton.roles, config.scm_params, config.dag_params, config.rscm, seed
    )
    arms = SUITES[suite]
    db_id = f"db-{seed:08d}"
    case = {"seed": seed, "id": db_id, "arms": {}}
    for arm, (global_dag, quantile, direct) in arms.items():
        if suite == "holdout":
            global_dag = bool(seed % 2)
            quantile = bool((seed // 2) % 2)
        alpha = (1.0, 5.0) if quantile else None
        params = replace(
            config.rscm,
            categorical_alpha_choices=alpha,
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
            frames, causal = local_frames(dataset, tables, skeleton, alpha)
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
                    with scoped_rng(keyed_seed(seed, "experiment-missingness", name, col)):
                        share = config.database_params.column_nan_perc.sample_uniform()
                        absent = np.random.choice(
                            len(frame), int(share * len(frame)), replace=False
                        )
                    frame.loc[absent, col] = np.nan
                    kind = "reg"
                if kind and frame[col].nunique(dropna=True) > 1:
                    task_list.append([db_id, name, col, [], kind])
            # Ensure PK-only rows still get physical identities in the paper preprocessor.
            if not frame[list(features(info))].notna().any(axis=1).all():
                frame["_rt_row"] = f"row of {name}"
            database[name] = Table(df=frame, pkey_col=pk, fkey_col_to_pkey_table=fks, time_col=None)
        if not task_list:
            return {"seed": seed, "rejected": f"no varying numeric or Boolean targets in {arm}"}
        dest = stage / arm / db_id
        dest.mkdir(parents=True)
        Database(database).save(dest / "db")
        write_json(dest / "causal.json", causal)
        case["arms"][arm] = {
            "global_dag": global_dag,
            "quantiles": quantile,
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
                    "generation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "databases": entries,
                    "tasks": tasks,
                },
            )
        stage.rename(output)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthesize", "export"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if (args.suite == "holdout") != (args.split == "val"):
        parser.error("Use holdout with val and factorial/sampler with train")
    root = args.data_root.expanduser().resolve()
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
