"""
Generate synthetic tabular data using Structured Causal Models.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_frame import stype
from tqdm import tqdm

from relrel.config import DAGParams, SCMParams
from relrel.dag import DAG_REGISTRY
from relrel.generation import FixedMLP, calibrate_quantiles
from relrel.rng import keyed_seed
from relrel.ts import CategoricalTSDataGenerator, TSDataGenerator
from relrel.utils import (
    MLP,
    CategoricalDecoder,
    CategoricalEncoder,
    TableType,
    get_bipartite_hsbm,
    set_random_seed,
)


class SCM:
    def __init__(
        self,
        table_name: str,
        child_table_names: list[str],
        feature_columns: dict[str, stype],
        pkey_col: str,
        fkey_col_to_pkey_table: dict[str, str],
        foreign_scm_info: dict[str, SCM],
        scm_params: SCMParams,
        dag_params: DAGParams,
        seed: int | None = None,
        *,
        categorical_alpha_choices: tuple[float, ...] | None = None,
        fixed_mlp: bool = False,
        quantile_seed: int | None = None,
    ):
        self.table_name = table_name
        self.child_table_names = child_table_names
        self.num_col_nodes = len(feature_columns)
        self.feature_columns = feature_columns
        self.pkey_col = pkey_col
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table
        self.foreign_scm_info = foreign_scm_info
        self.scm_params = scm_params
        self.dag_params = dag_params
        self.seed = seed
        self.categorical_alpha_choices = categorical_alpha_choices
        self.fixed_mlp = fixed_mlp
        self.quantile_seed = quantile_seed
        self.category_calibrations: dict[str, dict] = {}
        if self.seed:
            set_random_seed(self.seed)

        self.validate_foreign_scms()
        self.initialize_dag()
        self.initialize_nodes_and_edges()

    def initialize_dag(self):
        dag_class = self.scm_params.scm_layout_choices.sample_uniform()
        dag_class = DAG_REGISTRY[dag_class]
        scm_col_node_perc = self.scm_params.scm_col_node_perc_choices.sample_uniform()
        num_nodes = (
            self.num_col_nodes
            + int(np.ceil(self.num_col_nodes / scm_col_node_perc))
            + 3  # buffer to avoid empty graphs
        )
        self.dag = dag_class(num_nodes=num_nodes, dag_params=self.dag_params, seed=self.seed)
        self.num_edges = len(self.dag.graph.edges)

    def validate_foreign_scms(self):
        for foreign_table_name, scm in self.foreign_scm_info.items():
            assert hasattr(scm, "df"), (
                f"the foreign_scm for table: {foreign_table_name} does not contain the `self.df` attribute"
            )

    def _get_categorical_encoder(self, num_categories: int):
        return CategoricalEncoder(
            scm_params=self.scm_params,
            num_embeddings=num_categories,
            embedding_dim=self.scm_params.mlp_emb_dim,
        )

    def _get_categorical_decoder(self, num_categories: int):
        return CategoricalDecoder(
            scm_params=self.scm_params,
            num_embeddings=num_categories,
            embedding_dim=self.scm_params.mlp_emb_dim,
        )

    def _get_numerical_encoder(self):
        return MLP(
            scm_params=self.scm_params,
            in_dim=self.scm_params.mlp_in_dim,
            hid_dim=self.scm_params.mlp_emb_dim,
            out_dim=self.scm_params.mlp_emb_dim,
        )

    def _get_numerical_decoder(self):
        return MLP(
            scm_params=self.scm_params,
            in_dim=self.scm_params.mlp_emb_dim,
            hid_dim=self.scm_params.mlp_emb_dim,
            out_dim=self.scm_params.mlp_out_dim,
        )

    def _maybe_fix(self, projection):
        if not self.fixed_mlp:
            return projection
        if isinstance(projection, MLP):
            return FixedMLP(projection, self.scm_params)
        projection.mlp = FixedMLP(projection.mlp, self.scm_params)
        return projection

    def get_encoder(self, _stype: stype, num_categories: int | None):
        if _stype == stype.numerical:
            return self._maybe_fix(self._get_numerical_encoder())
        elif _stype == stype.categorical:
            return self._maybe_fix(self._get_categorical_encoder(num_categories=num_categories))

    def get_decoder(self, _stype: stype, num_categories: int | None):
        if _stype == stype.numerical or self.categorical_alpha_choices is not None:
            return self._maybe_fix(self._get_numerical_decoder())
        elif _stype == stype.categorical:
            return self._maybe_fix(self._get_categorical_decoder(num_categories=num_categories))

    def _get_ts_data_gen(self, num_rows: int, table_type: TableType):
        num_points = num_rows
        min_value, max_value = sorted(np.random.uniform(size=2))
        trend_alpha = self.scm_params.ts_trend_alpha_choices.sample_uniform()
        trend_scale = (
            self.scm_params.activity_table_ts_trend_scale_choices.sample_uniform()
            if table_type == TableType.Activity
            else self.scm_params.entity_table_ts_trend_scale
        )
        noise_scale = (
            self.scm_params.activity_table_ts_noise_scale
            if table_type == TableType.Activity
            else self.scm_params.entity_table_ts_noise_scale
        )
        cycle_frequency_perc = self.scm_params.ts_cycle_freq_perc_choices.sample_uniform()
        cycle_scale = (
            self.scm_params.activity_table_ts_cycle_scale_choices.sample_uniform()
            if table_type == TableType.Activity
            else self.scm_params.entity_table_ts_cycle_scale
        )
        return TSDataGenerator(
            num_points=num_points,
            min_value=min_value,
            max_value=max_value,
            trend_alpha=trend_alpha,
            trend_scale=trend_scale,
            cycle_frequency=np.ceil(num_rows * cycle_frequency_perc),
            cycle_scale=cycle_scale,
            noise_scale=noise_scale,
        )

    def initialize_ts_data_gens(self, num_rows: int, table_type: TableType):
        self.source_node_to_ts_data_gen = {}
        for node in self.source_nodes:
            _stype = self.dag.graph.nodes[node]["_stype"]
            if _stype == stype.numerical or self.categorical_alpha_choices is not None:
                self.source_node_to_ts_data_gen[node] = self._get_ts_data_gen(
                    num_rows=num_rows, table_type=table_type
                )
            elif _stype == stype.categorical:
                num_categories = self.dag.graph.nodes[node]["num_categories"]
                self.source_node_to_ts_data_gen[node] = CategoricalTSDataGenerator(
                    ts_data_gens=[
                        self._get_ts_data_gen(num_rows=num_rows, table_type=table_type)
                        for _ in range(num_categories)
                    ]
                )

    def initialize_nodes_and_edges(self):
        self.source_nodes = [
            node for node in self.dag.graph.nodes if self.dag.graph.in_degree(node) == 0
        ]
        self.col_nodes = np.random.choice(
            a=self.dag.graph.nodes, size=self.num_col_nodes, replace=False
        )
        for col_idx, (col_name, col_info) in enumerate(self.feature_columns.items()):
            col_node = self.col_nodes[col_idx]
            self.dag.graph.nodes[col_node]["col_name"] = col_name
            self.dag.graph.nodes[col_node]["_stype"] = col_info["_stype"]
            self.dag.graph.nodes[col_node]["num_categories"] = (
                len(col_info["categories"]) if col_info["categories"] else None
            )

        self._reset_node_attributes()
        self._reset_edge_attributes()

    def _reset_node_attributes(self):
        for node in self.dag.graph.nodes:
            if node in self.col_nodes:
                _stype = self.dag.graph.nodes[node]["_stype"]
                num_categories = self.dag.graph.nodes[node]["num_categories"]
            else:
                _stype = self.scm_params.col_stype_choices.sample_uniform()
                self.dag.graph.nodes[node]["_stype"] = _stype
                if _stype == stype.categorical:
                    num_categories = self.scm_params.num_categories_choices.sample_uniform()
                else:
                    num_categories = None
                self.dag.graph.nodes[node]["num_categories"] = num_categories

            self.dag.graph.nodes[node]["noise_dist"] = (
                self.scm_params.node_noise_dist_choices.sample_uniform()
            )
            self.dag.graph.nodes[node]["decoder"] = self.get_decoder(
                _stype=_stype, num_categories=num_categories
            )
            if node in self.col_nodes:
                self.dag.graph.nodes[node]["collation_encoders"] = {}
                for child_table_name in self.child_table_names:
                    self.dag.graph.nodes[node]["collation_encoders"][
                        (self.table_name, child_table_name)
                    ] = self.get_encoder(_stype=_stype, num_categories=num_categories)

    def _reset_edge_attributes(self):
        for parent_node, child_node in self.dag.graph.edges:
            parent_node_stype = self.dag.graph.nodes[parent_node]["_stype"]
            parent_node_num_categories = self.dag.graph.nodes[parent_node]["num_categories"]
            self.dag.graph.edges[parent_node, child_node]["encoder"] = self.get_encoder(
                _stype=parent_node_stype, num_categories=parent_node_num_categories
            )

    def propagate(self, row_idx: int, foreign_row_idxs: list[int], foreign_scms: list[SCM]):
        foreign_scms_row_embds: list[list] = []
        for foreign_row_idx, foreign_scm in zip(foreign_row_idxs, foreign_scms):
            foreign_row_embds = foreign_scm.collate_feature_embeddings(
                row_idx=foreign_row_idx, child_table_name=self.table_name
            )
            foreign_scms_row_embds.append(foreign_row_embds)

        topological_gens = nx.topological_generations(self.dag.graph)
        edge_idx = 0
        for gen in topological_gens:
            for node in gen:
                node_stype = self.dag.graph.nodes[node]["_stype"]
                if node in self.source_nodes:
                    value = self.source_node_to_ts_data_gen[node].get_value(row_idx=row_idx)
                    if node_stype == stype.categorical:
                        value = torch.LongTensor([value])
                    else:
                        value = torch.Tensor([value])
                    self.dag.graph.nodes[node]["value"] = value
                else:
                    parent_nodes = self.dag.graph.predecessors(node)
                    node_num_categories = self.dag.graph.nodes[node]["num_categories"]

                    # directly add noise
                    noise_dist = self.dag.graph.nodes[node]["noise_dist"]
                    node_emb = (
                        noise_dist.sample(sample_shape=(self.scm_params.mlp_emb_dim,)).squeeze()
                        / self.scm_params.mlp_emb_dim
                    )
                    for parent_node in parent_nodes:
                        parent_attrs = self.dag.graph.nodes[parent_node]
                        encoder = self.dag.graph.edges[parent_node, node]["encoder"]
                        parent_emb = encoder(parent_attrs["value"]).squeeze()
                        weight = self.dag.graph.get_edge_data(parent_node, node)["weight"]
                        node_emb += weight * parent_emb

                    for foreign_row_embds in foreign_scms_row_embds:
                        for foreign_row_embd in foreign_row_embds:
                            weight = 1 / len(foreign_row_embds)
                            node_emb += weight * foreign_row_embd

                    decoder = self.dag.graph.nodes[node]["decoder"]
                    value = decoder(node_emb)
                    self.dag.graph.nodes[node]["value"] = value
                    edge_idx += 1

    def generate_row(self, row_idx: int):
        row = {self.pkey_col: row_idx} if self.pkey_col is not None else {}
        foreign_row_idxs = []
        foreign_scms = []
        for fkey_col, foreign_table_name in self.fkey_col_to_pkey_table.items():
            foreign_scm = self.foreign_scm_info[foreign_table_name]
            foreign_scms.append(foreign_scm)
            bi_g = self.bi_fk_pk_graph_map[foreign_table_name]
            parent_node_name = list(bi_g.in_edges(f"b{row_idx}"))[0][0]
            foreign_row_idx = bi_g.nodes[parent_node_name]["node_idx"]
            foreign_row_idxs.append(foreign_row_idx)
            row[fkey_col] = foreign_row_idx

        self.propagate(
            row_idx=row_idx,
            foreign_row_idxs=foreign_row_idxs,
            foreign_scms=foreign_scms,
        )
        for idx, node in enumerate(sorted(self.col_nodes)):
            col_name = self.dag.graph.nodes[node]["col_name"]
            row[col_name] = self.dag.graph.nodes[node]["value"].item()
        return row

    def initialize_bi_fk_pk_graph_map(self):
        self.bi_fk_pk_graph_map = {}
        for foreign_table_name, foreign_scm in tqdm(
            self.foreign_scm_info.items(),
            desc="Generating bi_fk_pk_graphs",
            leave=False,
        ):
            num_levels = self.scm_params.bi_hsbm_levels_choices.sample_uniform()
            hierarchy_a = [
                self.scm_params.bi_hsbm_clusters_per_level_choices.sample_uniform()
                for _ in range(num_levels)
            ]
            hierarchy_b = [
                self.scm_params.bi_hsbm_clusters_per_level_choices.sample_uniform()
                for _ in range(num_levels)
            ]
            bi_g = get_bipartite_hsbm(
                size_a=len(foreign_scm.df),
                size_b=self.num_rows,
                hierarchy_a=hierarchy_a,
                hierarchy_b=hierarchy_b,
            )
            self.bi_fk_pk_graph_map[foreign_table_name] = bi_g

    def _apply_projection(self, projection, tensor: torch.Tensor) -> torch.Tensor:
        if self.fixed_mlp:
            return projection(tensor)
        return torch.stack([projection(tensor[i]) for i in range(len(tensor))], dim=0)

    @torch.no_grad()
    def _generate_df_quantile(
        self,
        num_rows: int,
        min_timestamp: pd.Timestamp | None = None,
        max_timestamp: pd.Timestamp | None = None,
    ):
        """Evaluate PluRel's per-table SCM node-by-node for eager quantile decoding."""
        frame_data: dict[str, np.ndarray] = {}
        if self.pkey_col is not None:
            frame_data[self.pkey_col] = np.arange(num_rows, dtype=np.int64)

        foreign_input = torch.zeros((num_rows, self.scm_params.mlp_emb_dim))
        for fkey_col, foreign_table_name in self.fkey_col_to_pkey_table.items():
            foreign_scm = self.foreign_scm_info[foreign_table_name]
            bi_g = self.bi_fk_pk_graph_map[foreign_table_name]
            offsets = np.array(
                [
                    bi_g.nodes[list(bi_g.in_edges(f"b{row_idx}"))[0][0]]["node_idx"]
                    for row_idx in range(num_rows)
                ],
                dtype=np.int64,
            )
            frame_data[fkey_col] = offsets
            parent_col_nodes = sorted(foreign_scm.col_nodes)
            if not parent_col_nodes:
                continue
            weight = 1.0 / len(parent_col_nodes)
            for p_node in parent_col_nodes:
                p_attrs = foreign_scm.dag.graph.nodes[p_node]
                col_name = p_attrs["col_name"]
                encoder = p_attrs["collation_encoders"][(foreign_table_name, self.table_name)]
                parent_vals = torch.from_numpy(foreign_scm.df[col_name].to_numpy(copy=True))
                if p_attrs["_stype"] == stype.numerical:
                    parent_vals = parent_vals.float().reshape(-1, 1)
                else:
                    parent_vals = parent_vals.long()
                gathered = parent_vals[offsets]
                foreign_input += weight * self._apply_projection(encoder, gathered).reshape(
                    num_rows, self.scm_params.mlp_emb_dim
                )

        node_values: dict[int, torch.Tensor] = {}
        self.category_calibrations = {}
        for gen in nx.topological_generations(self.dag.graph):
            for node in gen:
                attrs = self.dag.graph.nodes[node]
                node_stype = attrs["_stype"]
                if node in self.source_nodes:
                    raw = [
                        self.source_node_to_ts_data_gen[node].get_value(row_idx=r)
                        for r in range(num_rows)
                    ]
                    value = torch.tensor(raw, dtype=torch.float32)
                else:
                    noise_dist = attrs["noise_dist"]
                    node_emb = (
                        noise_dist.sample(
                            sample_shape=(num_rows, self.scm_params.mlp_emb_dim)
                        ).reshape(num_rows, self.scm_params.mlp_emb_dim)
                        / self.scm_params.mlp_emb_dim
                    )
                    for parent_node in self.dag.graph.predecessors(node):
                        encoder = self.dag.graph.edges[parent_node, node]["encoder"]
                        weight = self.dag.graph.get_edge_data(parent_node, node)["weight"]
                        parent_emb = self._apply_projection(
                            encoder, node_values[parent_node]
                        ).reshape(num_rows, self.scm_params.mlp_emb_dim)
                        node_emb += weight * parent_emb
                    node_emb = node_emb + foreign_input
                    decoder = attrs["decoder"]
                    value = self._apply_projection(decoder, node_emb).reshape(num_rows)

                if node_stype == stype.categorical:
                    num_categories = attrs["num_categories"]
                    if self.quantile_seed is not None:
                        q_seed = keyed_seed(
                            self.quantile_seed, "local-quantile", self.table_name, int(node)
                        )
                    else:
                        q_seed = int(np.random.randint(0, 1 << 31))
                    rng = np.random.default_rng(q_seed)
                    value, calibration = calibrate_quantiles(
                        value, num_categories, self.categorical_alpha_choices, rng
                    )
                    key = attrs.get("col_name", f"latent:{node}")
                    self.category_calibrations[key] = calibration
                    node_values[node] = value.long()
                else:
                    node_values[node] = value.float().reshape(num_rows, 1)

        for node in sorted(self.col_nodes):
            col_name = self.dag.graph.nodes[node]["col_name"]
            if self.dag.graph.nodes[node]["_stype"] == stype.categorical:
                frame_data[col_name] = node_values[node].cpu().numpy().astype(np.int64)
            else:
                frame_data[col_name] = (
                    node_values[node].reshape(-1).cpu().numpy().astype(np.float64)
                )

        self.df = pd.DataFrame(frame_data)
        if min_timestamp and max_timestamp:
            self.df["date"] = pd.date_range(
                start=min_timestamp, end=max_timestamp, periods=num_rows
            )
        return self.df

    def generate_df(
        self,
        num_rows: int,
        table_type: TableType,
        min_timestamp: pd.Timestamp | None = None,
        max_timestamp: pd.Timestamp | None = None,
    ):
        self.num_rows = num_rows
        self.initialize_ts_data_gens(num_rows=num_rows, table_type=table_type)
        self.initialize_bi_fk_pk_graph_map()
        if self.categorical_alpha_choices is not None:
            return self._generate_df_quantile(
                num_rows=num_rows,
                min_timestamp=min_timestamp,
                max_timestamp=max_timestamp,
            )
        self.df = pd.DataFrame(
            [
                self.generate_row(row_idx=row_idx)
                for row_idx in tqdm(range(self.num_rows), desc="generating rows", leave=False)
            ]
        )
        if min_timestamp and max_timestamp:
            self.df["date"] = pd.date_range(
                start=min_timestamp, end=max_timestamp, periods=num_rows
            )
        return self.df

    def collate_feature_embeddings(self, row_idx: int, child_table_name: int):
        col_to_stype = {}
        col_to_num_categories = {}
        col_to_collation_encoder = {}
        for f_idx, node in enumerate(sorted(self.col_nodes)):
            col_name = self.dag.graph.nodes[node]["col_name"]
            col_to_stype[col_name] = self.dag.graph.nodes[node]["_stype"]
            col_to_num_categories[col_name] = self.dag.graph.nodes[node]["num_categories"]
            col_to_collation_encoder[col_name] = self.dag.graph.nodes[node]["collation_encoders"][
                (self.table_name, child_table_name)
            ]
        row = self.df.iloc[row_idx].to_dict()
        row_embds = []
        num_cols = len(col_to_stype)
        # for p->f embedding propagation
        for col_name, value in row.items():
            if col_name not in col_to_stype:
                continue
            _stype = col_to_stype[col_name]
            if _stype == stype.numerical:
                value_tensor = torch.Tensor([value])
            elif _stype == stype.categorical:
                value_tensor = torch.LongTensor([value])
            num_categories = col_to_num_categories[col_name]
            encoder = col_to_collation_encoder[col_name]
            row_embds.append(encoder(value_tensor).squeeze())
        return row_embds
