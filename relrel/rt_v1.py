"""Prepare a separate RelBench export for the paper RT release."""

import json
import tempfile
from pathlib import Path

import numpy as np
from pandas.api.types import is_bool_dtype
from relbench.base import Database


def export_rt_v1(source: Path, destination: Path):
    """Export tables and training task tuples for unmodified RT at tag rt-v1."""
    if destination.exists():
        raise FileExistsError(f"RT export destination already exists: {destination}")
    features_by_table = json.loads((source / "causal.json").read_text())["tables"]
    database = Database.load(source / "db")
    if set(database.table_dict) != set(features_by_table):
        raise ValueError("RelRel tables and causal metadata disagree")
    tasks = []
    for name, table in sorted(database.table_dict.items()):
        if name.count("-") == 2:
            raise ValueError(
                f"RT v1 would skip table {name!r} because its name contains two hyphens"
            )
        if len(table.fkey_col_to_pkey_table) > 5:
            raise ValueError(f"RT v1 accepts at most five foreign keys per table: {name!r}")
        if len(table) == 0:
            raise ValueError(f"RT v1 export requires nonempty tables: {name!r}")
        for column, feature in features_by_table[name]["features"].items():
            values = table.df[column]
            varying = values.nunique() > 1
            if feature["type"] == "categorical":
                if is_bool_dtype(values.dtype) and varying:
                    tasks.append((destination.name, name, column, []))
                elif not isinstance(feature["categories"][0], str):
                    # RT v1 treats integers as numbers and normalizes constant booleans by zero.
                    table.df[column] = values.map(
                        lambda value: f"{name}.{column} category {int(value)}"
                    )
            elif varying:
                tasks.append((destination.name, name, column, []))
        features = features_by_table[name]["features"]
        if not features or not table.df[list(features)].notna().any(axis=1).all():
            # RT v1 initializes a row's identity only when processing a feature or a nonnull FK.
            # A constant context token preserves rows that would otherwise be empty.
            column = "_rt_row"
            while column in table.df:
                column += "_"
            table.df[column] = "row of " + name
        # RT expects contiguous signed 64-bit IDs. RelBench already reindexed the source cache.
        if not np.array_equal(table.df[table.pkey_col].to_numpy(), np.arange(len(table))):
            raise ValueError(f"RT v1 requires contiguous primary keys in {name!r}")
        for column in table.fkey_col_to_pkey_table:
            table.df[column] = table.df[column].astype("Int64")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
        staging = Path(temp_dir) / destination.name
        database.save(staging / "db")
        (staging / "train_tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
        staging.rename(destination)
    return tasks
