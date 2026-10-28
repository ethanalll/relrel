"""Evaluate a relational SCM in global feature-column order, using v1 factories."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch_frame import stype

from relrel.config import Config
from relrel.relational import compile_grounding
from relrel.rng import canonical_json, config_fingerprint, keyed_seed, scoped_rng
from relrel.rscm import Column, ColumnGraph, FKRole
from relrel.skeleton import Skeleton, check_memory
from relrel.ts import CategoricalTSDataGenerator, TSDataGenerator
from relrel.utils import MLP, CategoricalDecoder, CategoricalEncoder, TableType

GENERATOR_VERSION = "v1.0.0-global-rscm-v1"
BASE_REVISION = "2a273cfd21933ee4893dfcb862a3edaed45ac665"


class FixedMLP:
    """Keep the release's sampled weights; draw each hidden activation once.

    The legacy MLP remains unchanged. Fixed functions make structural replay
    and interventions well defined without incidental per-call activation draws.
    """

    def __init__(self, mlp: MLP, scm_params):
        self.weights = mlp.weights
        self.activations = [
            scm_params.activation_choices.sample_uniform() for _ in self.weights[:-1]
        ]

    def __call__(self, x):
        for weight, activation in zip(self.weights[:-1], self.activations):
            x = activation(x @ weight)
        return x @ self.weights[-1]


def make_projection(scm_params, info: dict, *, decoder: bool):
    """Adapt the original numerical/categorical projection classes."""
    if info["_stype"] == stype.numerical:
        mlp = MLP(
            scm_params,
            scm_params.mlp_emb_dim if decoder else scm_params.mlp_in_dim,
            scm_params.mlp_emb_dim,
            scm_params.mlp_out_dim if decoder else scm_params.mlp_emb_dim,
        )
        return FixedMLP(mlp, scm_params)
    if info["_stype"] != stype.categorical or not info.get("categories"):
        raise ValueError("RSCM features must be numerical or have an explicit category inventory")
    cls = CategoricalDecoder if decoder else CategoricalEncoder
    function = cls(scm_params, len(info["categories"]), scm_params.mlp_emb_dim)
    function.mlp = FixedMLP(function.mlp, scm_params)
    return function


def make_source(scm_params, info: dict, rows: int, table_type: TableType):
    """Use the release's clipped temporal processes, including entity scales."""

    def numerical():
        low, high = sorted(np.random.uniform(size=2))
        alpha = scm_params.ts_trend_alpha_choices.sample_uniform()
        activity = table_type == TableType.Activity
        trend = (
            scm_params.activity_table_ts_trend_scale_choices.sample_uniform()
            if activity
            else scm_params.entity_table_ts_trend_scale
        )
        noise = (
            scm_params.activity_table_ts_noise_scale
            if activity
            else scm_params.entity_table_ts_noise_scale
        )
        fraction = scm_params.ts_cycle_freq_perc_choices.sample_uniform()
        cycle = (
            scm_params.activity_table_ts_cycle_scale_choices.sample_uniform()
            if activity
            else scm_params.entity_table_ts_cycle_scale
        )
        return TSDataGenerator(
            rows, low, high, alpha, trend, np.ceil(rows * fraction), cycle, noise
        )

    if info["_stype"] == stype.numerical:
        return numerical()
    return CategoricalTSDataGenerator([numerical() for _ in info["categories"]])


class GlobalSCMGenerator:
    """One shared mechanism per column; FK paths select row-level inputs.

    Graph rank strictly increases along every grounded cell edge, including
    reverse FK paths and table-return chains such as A.x -> B.y -> A.z.
    A zero-parent row of a non-root column keeps its decoder and explicit noise.
    """

    def __init__(
        self,
        tables: dict[str, dict],
        graph: ColumnGraph,
        skeleton: Skeleton,
        config: Config,
        seed: int = 0,
    ):
        self.tables, self.graph, self.skeleton = tables, graph, skeleton
        self.config, self.seed = config, int(seed)
        if config.scm_params.mlp_in_dim != 1 or config.scm_params.mlp_out_dim != 1:
            raise ValueError("RSCM cells require scalar numerical inputs and outputs")
        if type(config.scm_params.mlp_emb_dim) is not int or config.scm_params.mlp_emb_dim < 1:
            raise ValueError("RSCM embedding width must be a positive integer")
        if set(tables) != set(skeleton.row_counts):
            raise ValueError("Table inventory and skeleton must agree")
        expected = {
            Column(name, column)
            for name, info in tables.items()
            for column in info["columns"]
            if column != info["pkey_col"] and column not in info["fkey_col_to_pkey_table"]
        }
        if expected != set(graph.columns):
            raise ValueError("Column graph must contain every qualified feature exactly once")
        expected_roles = {
            FKRole(foreign_key, name, parent)
            for name, info in tables.items()
            for foreign_key, parent in info["fkey_col_to_pkey_table"].items()
        }
        if expected_roles != set(skeleton.roles):
            raise ValueError("Skeleton FK inventory must match the declared schema")
        for column in graph.columns:
            info = self._info(column)
            if info["_stype"] not in (stype.numerical, stype.categorical):
                raise ValueError("RSCM features must be numerical or categorical")
            if info["_stype"] == stype.categorical and not info.get("categories"):
                raise ValueError("Categorical RSCM features require an explicit category inventory")
        for name, info in tables.items():
            if not info["pkey_col"] or info["pkey_col"] not in info["columns"]:
                raise ValueError(f"RSCM table {name} requires a scalar primary key")
        graph.validate(skeleton.roles, config.rscm.max_hops, config.rscm.column_parent_cap)
        self.encoders, self.decoders, self.noise_distributions, self.plans = {}, {}, {}, {}
        self.function_keys = {}
        self.raw_values: dict[Column, torch.Tensor] = {}
        limits = config.rscm
        index_bytes = skeleton.nbytes
        for edge in graph.edges:
            plan = compile_grounding(skeleton, edge, limits.index_memory_bytes - index_bytes)
            self.plans[edge.identity] = plan
            index_bytes += plan.nbytes
            with scoped_rng(self._key("encoder", edge.identity)):
                self.encoders[edge.identity] = make_projection(
                    config.scm_params, self._info(edge.source), decoder=False
                )
        for column in graph.columns:
            if not graph.incoming(column):
                continue
            with scoped_rng(self._key("decoder", column.identity)):
                self.decoders[column] = make_projection(
                    config.scm_params, self._info(column), decoder=True
                )
            with scoped_rng(self._key("noise-family", column.identity)):
                self.noise_distributions[column] = (
                    config.scm_params.node_noise_dist_choices.sample_uniform()
                )

    def _info(self, column):
        return self.tables[column.table]["columns"][column.name]

    def _key(self, kind, identity):
        key = keyed_seed(self.seed, GENERATOR_VERSION, kind, identity)
        self.function_keys[f"{kind}:{identity!r}"] = key
        return key

    def _source(self, column):
        count = self.skeleton.row_counts[column.table]
        with scoped_rng(self._key("source", column.identity)):
            source = make_source(
                self.config.scm_params,
                self._info(column),
                count,
                self.tables[column.table].get("type", TableType.Entity),
            )
            values = [source.get_value(row) for row in range(count)]
        dtype = torch.long if self._info(column)["_stype"] == stype.categorical else torch.float32
        stable = torch.tensor(values, dtype=dtype)
        result = torch.empty_like(stable)
        result[np.argsort(self.skeleton.row_ids[column.table], kind="stable")] = stable
        return result

    def _noise(self, column):
        count, width = self.skeleton.row_counts[column.table], self.config.scm_params.mlp_emb_dim
        with scoped_rng(self._key("noise", column.identity)):
            values = self.noise_distributions[column].sample((count, width)).reshape(count, width)
        result = torch.empty_like(values)
        result[np.argsort(self.skeleton.row_ids[column.table], kind="stable")] = values / width
        return result

    def _intervention(self, column, values):
        array = np.asarray(values)
        if array.shape != (self.skeleton.row_counts[column.table],) or not np.isfinite(array).all():
            raise ValueError("An intervention must specify one finite value per physical row")
        if self._info(column)["_stype"] == stype.categorical:
            categories = len(self._info(column)["categories"])
            if np.any(array != np.floor(array)) or np.any((array < 0) | (array >= categories)):
                raise ValueError("Categorical interventions must use valid category indices")
            return torch.tensor(array.copy(), dtype=torch.long)
        return torch.tensor(array.copy(), dtype=torch.float32)

    @torch.no_grad()
    def evaluate(self, interventions: dict[Column, np.ndarray] | None = None):
        """Return pre-export frames; repeat calls reuse functions and exogenous draws."""
        interventions = interventions or {}
        if set(interventions) - set(self.graph.columns):
            raise ValueError("Intervention references an unknown feature column")
        limits, width = self.config.rscm, self.config.scm_params.mlp_emb_dim
        self.raw_values = {}
        for column in self.graph.order:
            count = self.skeleton.row_counts[column.table]
            check_memory(count * width * 4, limits.temp_memory_bytes, "Column latent buffer")
            if column in interventions:
                values = self._intervention(column, interventions[column])
            elif not self.graph.incoming(column):
                values = self._source(column)
            else:
                latent = self._noise(column)
                for edge in self.graph.incoming(column):
                    source = self.raw_values[edge.source]
                    check_memory(
                        len(source) * width * 4,
                        limits.temp_memory_bytes,
                        "Encoded source column",
                    )
                    if self._info(edge.source)["_stype"] == stype.numerical:
                        source = source.unsqueeze(-1)
                    encoded = self.encoders[edge.identity](source)
                    pool = self.plans[edge.identity].prepare_pool(
                        encoded,
                        policy=limits.parent_policy,
                        cap=limits.cell_parent_cap_per_edge,
                        seed=self.seed,
                        edge_key=edge.identity,
                        pooling=edge.pooling,
                        temp_memory_bytes=limits.temp_memory_bytes,
                        parent_map_memory_bytes=limits.parent_map_memory_bytes,
                    )
                    for start in range(0, count, limits.batch_size):
                        stop = min(count, start + limits.batch_size)
                        latent[start:stop] += float(edge.weight) * pool.gather(slice(start, stop))
                dtype = (
                    torch.long
                    if self._info(column)["_stype"] == stype.categorical
                    else torch.float32
                )
                values = torch.empty(count, dtype=dtype)
                for start in range(0, count, limits.batch_size):
                    stop = min(count, start + limits.batch_size)
                    values[start:stop] = self.decoders[column](latent[start:stop]).reshape(-1)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite RSCM output in {column.qualified_name}")
            self.raw_values[column] = values
        frames = {}
        for name, info in self.tables.items():
            frame = {info["pkey_col"]: np.arange(self.skeleton.row_counts[name], dtype=np.int64)}
            for role in self.skeleton.roles:
                if role.child_table == name:
                    values = self.skeleton.forward(role)
                    frame[role.name] = pd.array(
                        [None if value == -1 else int(value) for value in values], dtype="Int64"
                    )
            for column in self.graph.columns:
                if column.table == name:
                    values = self.raw_values[column].cpu().numpy()
                    # v1 postprocessing identifies numerical/categorical columns by dtype.
                    frame[column.name] = values.astype(
                        np.int64
                        if self._info(column)["_stype"] == stype.categorical
                        else np.float64
                    )
            frames[name] = pd.DataFrame(frame)
        return frames

    def sidecar(self):
        return {
            "generator_version": GENERATOR_VERSION,
            "base_revision": BASE_REVISION,
            "seed": self.seed,
            "column_graph": self.graph.to_dict(),
            "row_counts": self.skeleton.row_counts,
            "skeleton_fingerprint": self.skeleton.fingerprint,
            "settings": asdict(self.config.rscm),
            "scm_settings": json.loads(canonical_json(self.config.scm_params)),
            "tables": {
                name: {
                    "pkey_col": info["pkey_col"],
                    "fkey_col_to_pkey_table": info["fkey_col_to_pkey_table"],
                    "type": info.get("type", TableType.Entity).value,
                    "features": {
                        column.name: {
                            "type": self._info(column)["_stype"].name,
                            "categories": self._info(column).get("categories"),
                        }
                        for column in self.graph.columns
                        if column.table == name
                    },
                }
                for name, info in self.tables.items()
            },
            "config_fingerprint": config_fingerprint(self.config.scm_params),
            "function_keys": self.function_keys,
            "mechanisms": "v1 MLPs and categorical embeddings with frozen hidden activations",
            "causal_values": "before v1 missingness and categorical export postprocessing",
            "temporal_policy": "static tables; source processes may vary with row index",
        }
