"""Per-table SCM evaluation for matched synthetic corpora."""

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_frame import stype

from relrel.generation import (
    CategoricalDecoding,
    FixedMLP,
    make_projection,
    make_source,
    mechanism_seed,
)
from relrel.rng import config_fingerprint, keyed_seed, scoped_rng
from relrel.rscm import Column
from relrel.scm import SCM
from relrel.utils import MLP


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


@torch.no_grad()
def local_frames(dataset, tables, skeleton, decoding: CategoricalDecoding):
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
            quantile = decoding.uses_quantiles(attrs)
            kind = {"_stype": attrs["_stype"], "categories": None}
            if not numerical:
                kind["categories"] = list(range(attrs["num_categories"]))
            column_name = attrs.get("col_name")
            identity = (
                Column(name, column_name).identity
                if column_name is not None
                else (name, "latent", int(node))
            )
            if graph.in_degree(node) == 0:
                with scoped_rng(mechanism_seed(seed, "source", identity)):
                    source = make_source(params, kind, n, info["type"], scalar_output=quantile)
                    value = torch.tensor(
                        [source.get_value(row) for row in range(n)],
                        dtype=torch.float32 if numerical or quantile else torch.long,
                    )
            else:
                with scoped_rng(mechanism_seed(seed, "noise-family", identity)):
                    distribution = params.node_noise_dist_choices.sample_uniform()
                with scoped_rng(mechanism_seed(seed, "noise", identity)):
                    noise = distribution.sample((n, params.mlp_emb_dim)).reshape(n, -1)
                latent = foreign_input + noise / params.mlp_emb_dim
                for parent in graph.predecessors(node):
                    edge = graph.edges[parent, node]
                    latent += edge["weight"] * edge["encoder"](values[parent])
                with scoped_rng(mechanism_seed(seed, "decoder", identity)):
                    decoder = make_projection(params, kind, decoder=True, scalar_output=quantile)
                value = decoder(latent).reshape(-1)
            if quantile:
                rng = np.random.default_rng(mechanism_seed(seed, "category-quantiles", identity))
                value, calibration = decoding.calibrate(value, attrs["num_categories"], rng)
                calibrations[attrs.get("col_name", f"latent:{node}")] = calibration
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite feature in {name}, node {node}")
            values[node] = value.reshape(n, 1) if numerical else value
            if node in scm.col_nodes:
                frame[attrs["col_name"]] = value.numpy()

        scm.df = frame
        scms[name], frames[name] = scm, frame
        diagnostics[name] = {
            "family": type(scm.dag).__name__,
            "graph_sha256": config_fingerprint(
                {
                    "nodes": sorted(int(node) for node in graph.nodes),
                    "edges": sorted(
                        (int(parent), int(child), float(data["weight"]))
                        for parent, child, data in graph.edges(data=True)
                    ),
                }
            ),
            "local_edges": graph.number_of_edges(),
            "quantiles": calibrations,
        }
    return frames, {"local_scm": diagnostics}
