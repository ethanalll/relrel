import dataclasses
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from relbench.base import Database, Dataset, Table
from torch_frame import stype

from relrel.config import Config
from relrel.generation import GENERATOR_VERSION, GlobalSCMGenerator
from relrel.rng import config_fingerprint, keyed_seed, scoped_rng
from relrel.rscm import Column, FKRole, sample_column_graph
from relrel.schema import RandomSchemaGraphBuilder, SQLSchemaGraphBuilder
from relrel.skeleton import Skeleton
from relrel.utils import TableType, set_random_seed

CATEGORICAL_EXPORT_VERSION = 2


class SyntheticDataset(Dataset):
    """
    A synthetic relbench compatible dataset.

    Args:
        seed (int): Random seed for reproducibility of dataset generation.
        config (Config): Config object with constants and sampling choices.
    """

    def __init__(self, seed: int, config: Config):
        self.seed = seed
        self.config = config
        set_random_seed(self.seed)
        self.initialize_timestamps()
        cache = self.config.cache_dir
        if cache is not None:
            cache = str(Path(cache).expanduser())
        super().__init__(cache_dir=cache)

    def initialize_timestamps(self):
        start_timestamp = self.config.database_params.min_timestamp
        end_timestamp = self.config.database_params.max_timestamp
        total_days = (end_timestamp - start_timestamp).days

        days = np.sort(np.random.choice(total_days, 2, replace=False))
        self.min_timestamp = start_timestamp + pd.Timedelta(days=int(days[0]))
        self.max_timestamp = start_timestamp + pd.Timedelta(days=int(days[1]))

        timestamps = pd.date_range(start=self.min_timestamp, end=self.max_timestamp)
        val_start_idx = int(len(timestamps) * self.config.val_split)
        test_start_idx = int(len(timestamps) * self.config.test_split)
        self.val_timestamp = timestamps[val_start_idx]
        self.test_timestamp = timestamps[test_start_idx]

    def generation_identity(self):
        settings = dataclasses.replace(self.config, cache_dir=None, schema_file=None)
        if self.config.rscm.max_foreign_keys is None:
            # Keep the cache identity for existing datasets generated without a cap.
            settings = vars(settings).copy()
            settings["rscm"] = vars(self.config.rscm).copy()
            del settings["rscm"]["max_foreign_keys"]
        schema_digest = None
        if self.config.schema_file:
            schema_digest = hashlib.sha256(Path(self.config.schema_file).read_bytes()).hexdigest()
        identity = {
            "generator_version": GENERATOR_VERSION,
            "categorical_export_version": CATEGORICAL_EXPORT_VERSION,
            "generation_mode": self.config.generation_mode,
            "seed": self.seed,
            "settings_fingerprint": config_fingerprint(settings),
            "schema_digest": schema_digest,
        }
        return {**identity, "fingerprint": config_fingerprint(identity)}

    def get_db(self, upto_test_timestamp=True):
        """Load or generate a RelBench database with matching RSCM metadata."""
        cache = Path(self.cache_dir).expanduser() if self.cache_dir is not None else None
        marker = cache / "generation.json" if cache else None
        identity = self.generation_identity()
        if cache and (cache / "db").exists() and any((cache / "db").iterdir()):
            if not marker.exists():
                raise ValueError(
                    "Existing database has no RSCM identity; choose a new cache directory"
                )
            stored = json.loads(marker.read_text())
            if stored.get("categorical_export_version") != CATEGORICAL_EXPORT_VERSION:
                raise ValueError(
                    "RSCM cache uses an older categorical export; choose a new cache directory"
                )
            if stored.get("fingerprint") != identity["fingerprint"]:
                raise ValueError("RSCM cache configuration differs; choose a new cache directory")
            if not (cache / "causal.json").exists() or not (cache / "skeleton.npz").exists():
                raise ValueError("RSCM cache is incomplete: missing causal metadata or skeleton")
            self.causal_metadata = json.loads((cache / "causal.json").read_text())
        db = super().get_db(upto_test_timestamp)
        if cache and hasattr(self, "generator"):
            arrays, roles, row_ids = {}, [], {}
            for i, (name, values) in enumerate(sorted(self.skeleton.row_ids.items())):
                key = f"row_ids_{i}"
                arrays[key], row_ids[name] = values, key
            for i, role in enumerate(self.skeleton.roles):
                key = f"fk_{i}"
                arrays[key] = self.skeleton.forward(role)
                roles.append({**role.to_dict(), "array": key})
            np.savez_compressed(cache / "skeleton.npz", **arrays)
            self.causal_metadata["skeleton_archive"] = {
                "path": "skeleton.npz",
                "row_ids": row_ids,
                "foreign_keys": roles,
            }
            for path, value in ((cache / "causal.json", self.causal_metadata), (marker, identity)):
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(json.dumps(value, indent=2) + "\n")
                temporary.replace(path)
        return db

    def _get_random_dag_table_relationships(self, num_tables: int):
        """
        Each table will have the following attributes:
        ```py
        {
            "columns":dict[col_name -> {
                "stype": stype,
                "categories": list[str] | None
            }],
            "pkey_col": str | None,
            "fkey_col_to_pkey_table": dict[str, str],
        }
        ```
        """
        builder = RandomSchemaGraphBuilder(
            config=self.config, num_tables=num_tables, seed=self.seed
        )
        return builder.build_graph()

    def _get_schema_file_table_relationships(self, schema_file: str):
        builder = SQLSchemaGraphBuilder(sql_file=schema_file)
        builder.load_schema()
        for table in builder.metadata.tables.values():
            for fk in table.foreign_keys:
                if not fk.column.primary_key or len(fk.column.table.primary_key.columns) != 1:
                    raise ValueError("RSCM foreign keys must reference a scalar primary key")
        return builder.build_graph()

    def configure_table_relationships(
        self, num_tables: int | None = None, schema_file: str | None = None
    ) -> nx.DiGraph:
        """
        Define the primary -> foreign key relationships between tables as a DAG.

        Args:
            num_tables (int): Number of tables for the layout based on random DAGs.
            schema_file (str): A predefined SQL based schema file.
        """
        assert num_tables or schema_file, "either `num_tables` or `schema_file` must not be None"
        # higher preference to schema file
        if schema_file:
            table_relationships = self._get_schema_file_table_relationships(schema_file=schema_file)
        elif num_tables:
            table_relationships = self._get_random_dag_table_relationships(num_tables=num_tables)

        activity_tables = [
            table
            for table in table_relationships.nodes
            if table_relationships.out_degree(table) == 0
        ]
        for table in table_relationships.nodes:
            if table in activity_tables:
                table_type = TableType.Activity
            else:
                table_type = TableType.Entity

            table_relationships.nodes[table]["type"] = table_type
            table_relationships.nodes[table]["num_rows"] = self.get_num_rows(table_type=table_type)
        return table_relationships

    def get_num_rows(self, table_type):
        if table_type == TableType.Entity:
            num_rows = self.config.database_params.num_rows_entity_table_choices.sample_uniform()
        if table_type == TableType.Activity:
            num_rows = self.config.database_params.num_rows_activity_table_choices.sample_uniform()
        return num_rows

    def implant_nan(self, df, pkey_col, fkey_cols):
        nan_perc = self.config.database_params.column_nan_perc.sample_uniform()
        num_nan_cells = int(np.floor(nan_perc * len(df)))
        for col_name, _type in df.dtypes.items():
            if col_name not in [pkey_col, *fkey_cols] and _type in [float]:
                nan_cells_idx = np.random.choice(df.index, size=num_nan_cells, replace=False)
                df.loc[nan_cells_idx, col_name] = np.nan
        return df

    @staticmethod
    def process_categorical_data(df, feature_columns):
        """Format category labels while preserving every generated feature."""
        for name, info in feature_columns.items():
            if info["_stype"] != stype.categorical:
                continue
            categories = info["categories"]
            if categories == [0, 1]:
                df[name] = df[name].astype(bool)
            elif isinstance(categories[0], str):
                df[name] = df[name].map(lambda index: categories[index])
        return df

    def make_db(self) -> Database:
        with scoped_rng(keyed_seed(self.seed, "schema")):
            num_tables = self.config.database_params.num_tables_choices.sample_uniform()
            self.table_relationships = self.configure_table_relationships(
                num_tables=num_tables, schema_file=self.config.schema_file
            )
        tables = {info["name"]: info for _, info in self.table_relationships.nodes(data=True)}
        # A SQL table without a declared PK still needs a physical row identity.
        for info in tables.values():
            if info["pkey_col"] is None:
                name = "row_idx"
                while name in info["columns"]:
                    name += "_"
                info["pkey_col"] = name
                info["columns"][name] = {"_stype": stype.categorical, "categories": None}
        columns = [
            Column(name, col)
            for name, info in sorted(tables.items())
            for col in sorted(info["columns"])
            if col != info["pkey_col"] and col not in info["fkey_col_to_pkey_table"]
        ]
        roles = [
            FKRole(fk, name, parent)
            for name, info in sorted(tables.items())
            for fk, parent in sorted(info["fkey_col_to_pkey_table"].items())
        ]
        self.column_graph = sample_column_graph(
            columns,
            roles,
            self.config.scm_params,
            self.config.dag_params,
            self.config.rscm,
            self.seed,
        )
        self.skeleton = Skeleton.from_schema(tables, self.config, self.seed)
        self.generator = GlobalSCMGenerator(
            tables, self.column_graph, self.skeleton, self.config, self.seed
        )
        frames = self.generator.evaluate()
        result = {}
        for name, info in tables.items():
            pk, fks = info["pkey_col"], info["fkey_col_to_pkey_table"]
            features = {
                col: value for col, value in info["columns"].items() if col != pk and col not in fks
            }
            with scoped_rng(keyed_seed(self.seed, "postprocessing", name)):
                frame = self.implant_nan(frames[name], pk, list(fks))
                frame = self.process_categorical_data(frame, features)
            result[name] = Table(df=frame, pkey_col=pk, fkey_col_to_pkey_table=fks, time_col=None)
        self.causal_metadata = self.generator.sidecar()
        return Database(result)
