import itertools
import json
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from torch_frame import stype

from relrel import (
    Choices,
    Column,
    ColumnGraph,
    Config,
    DatabaseParams,
    EdgeSpec,
    FKRole,
    GlobalSCMGenerator,
    PathStep,
    RSCMParams,
    SCMParams,
    Skeleton,
    SyntheticDataset,
    compile_grounding,
    sample_column_graph,
)
from relrel.bipartite import sample_bipartite_assignments
from relrel.rng import config_fingerprint, scoped_rng
from relrel.utils import TableType, get_bipartite_hsbm


def tiny_config(**kwargs):
    return Config(
        generation_mode="global_rscm",
        database_params=DatabaseParams(
            num_tables_choices=Choices("set", [3]),
            num_cols_choices=Choices("range", [3, 3]),
            num_rows_entity_table_choices=Choices("set", [8]),
            num_rows_activity_table_choices=Choices("set", [12]),
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "directions",
    [
        (),
        ("fk_to_pk",),
        ("pk_to_fk",),
        *itertools.product(("fk_to_pk", "pk_to_fk"), repeat=2),
    ],
)
@pytest.mark.parametrize("pooling", ["mean", "sum"])
@pytest.mark.parametrize("policy", ["full", "bounded"])
def test_grounding_and_pooling_against_literal_row_scan(directions, pooling, policy):
    tables, counts, links, path = ["a", "b", "c"], {"a": 3, "b": 4, "c": 5}, {}, []
    for i, direction in enumerate(directions):
        start, end = tables[i : i + 2]
        child, parent = (start, end) if direction == "fk_to_pk" else (end, start)
        role = FKRole(f"fk_{i}", child, parent)
        values = np.arange(counts[child]) // 2 % counts[parent]
        values[-1] = -1
        links[role] = values
        path.append(PathStep(role, direction))
    source = tables[len(directions)]
    skeleton = Skeleton.from_arrays(counts, links)
    edge = EdgeSpec(Column(source, "x"), Column("a", "y"), tuple(path), pooling=pooling)
    plan = compile_grounding(skeleton, edge)
    encoded = torch.arange(counts[source] * 3, dtype=torch.float64).reshape(-1, 3)
    output = plan.pool(encoded, policy=policy, cap=2, seed=7)
    for row in range(counts["a"]):
        endpoints = {row}
        for step in path:
            assignment = links[step.role]
            if step.direction == "fk_to_pk":
                endpoints = {int(assignment[r]) for r in endpoints if assignment[r] >= 0}
            else:
                endpoints = {r for r, p in enumerate(assignment) if p in endpoints}
        assert set(plan.parents(row)) == endpoints
        selected = plan.selected_parents(row, policy, 2, 7)
        assert set(selected) <= endpoints
        assert (
            len(selected) == min(2, len(endpoints))
            if policy == "bounded"
            else len(selected) == len(endpoints)
        )
        expected = torch.zeros(3, dtype=torch.float64)
        if len(selected):
            expected = encoded[selected.copy()].sum(0)
            if pooling == "mean":
                expected /= len(selected)
        torch.testing.assert_close(output[row], expected)


def test_graph_locality_cap_isolates_and_serialization():
    columns = tuple(Column(t, "x") for t in ("a", "b", "c", "d"))
    roles = (FKRole("a_id", "b", "a"), FKRole("b_id", "c", "b"))
    proposal = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3)])
    graph = sample_column_graph(
        columns,
        roles,
        SCMParams(),
        Config().dag_params,
        RSCMParams(max_hops=1, column_parent_cap=1),
        0,
        proposal=proposal,
        column_assignment=columns,
    )
    assert len(graph.columns) == 4
    assert not graph.incoming(columns[3]) and not graph.outgoing(columns[3])
    assert all(len(graph.incoming(c)) <= 1 for c in columns)
    assert all(len(e.path) <= 1 for e in graph.edges)
    assert ColumnGraph.from_dict(json.loads(json.dumps(graph.to_dict()))) == graph
    with pytest.raises(ValueError, match="topological rank"):
        ColumnGraph(
            columns, (EdgeSpec(columns[1], columns[0], (PathStep(roles[0], "pk_to_fk"),)),), columns
        )


@pytest.mark.parametrize("size", [0, 1, 2])
def test_small_column_inventories(size):
    columns = [Column("a", f"x{i}") for i in range(size)]
    graph = sample_column_graph(columns, [], SCMParams(), Config().dag_params, RSCMParams(), 0)
    assert set(graph.columns) == set(columns)


@pytest.mark.parametrize("seed", range(4))
def test_indexed_fk_sampler_matches_original_hsbm(seed):
    with scoped_rng(seed):
        original = get_bipartite_hsbm(17, 23, [2, 3], [3, 2])
    with scoped_rng(seed):
        assignments = sample_bipartite_assignments(17, 23, [2, 3], [3, 2])
    expected = [
        original.nodes[next(iter(original.predecessors(f"b{i}")))]["node_idx"] for i in range(23)
    ]
    np.testing.assert_array_equal(assignments, expected)


class ZeroNoise:
    def sample(self, shape):
        return torch.zeros((*shape, 1))


def numeric_table(features, fk=None):
    columns = {name: {"_stype": stype.numerical, "categories": None} for name in features}
    for name in ("id", *(fk or {})):
        columns[name] = {"_stype": stype.categorical, "categories": None}
    return {
        "columns": columns,
        "pkey_col": "id",
        "fkey_col_to_pkey_table": fk or {},
        "type": TableType.Entity,
    }


@pytest.mark.parametrize("policy", ["full", "bounded"])
def test_table_return_order_and_intervention(policy):
    income, amount, score = Column("a", "income"), Column("b", "amount"), Column("a", "score")
    role = FKRole("a_id", "b", "a")
    edges = (
        EdgeSpec(income, amount, (PathStep(role, "fk_to_pk"),)),
        EdgeSpec(amount, score, (PathStep(role, "pk_to_fk"),)),
    )
    graph = ColumnGraph((income, amount, score), edges, (income, amount, score))
    skeleton = Skeleton.from_arrays({"a": 3, "b": 5}, {role: [0, 0, 1, 2, -1]})
    tables = {
        "a": numeric_table(["income", "score"]),
        "b": numeric_table(["amount"], {"a_id": "a"}),
    }
    config = tiny_config(rscm=RSCMParams(parent_policy=policy), scm_params=SCMParams(mlp_emb_dim=1))
    generator = GlobalSCMGenerator(tables, graph, skeleton, config, 0)
    generator.encoders = {edge.identity: lambda x: x for edge in edges}
    generator.decoders = {col: lambda x: x[:, 0] for col in (amount, score)}
    generator.noise_distributions = {col: ZeroNoise() for col in (amount, score)}
    frames = generator.evaluate({income: np.array([1.0, 2.0, 3.0])})
    np.testing.assert_array_equal(frames["b"]["amount"], [1, 1, 2, 3, 0])
    np.testing.assert_array_equal(frames["a"]["score"], [1, 2, 3])
    changed = generator.evaluate({income: np.array([10.0, 2.0, 3.0])})
    np.testing.assert_array_equal(changed["a"]["score"], [10, 2, 3])
    assert pd.isna(changed["b"]["a_id"].iloc[-1])
    with pytest.raises(ValueError, match="unknown"):
        generator.evaluate({Column("a", "missing"): np.zeros(3)})


def test_categorical_values_are_encoded_before_reverse_pooling():
    source, target = Column("b", "category"), Column("a", "score")
    role = FKRole("a_id", "b", "a")
    edge = EdgeSpec(source, target, (PathStep(role, "pk_to_fk"),))
    tables = {"a": numeric_table(["score"]), "b": numeric_table(["category"], {"a_id": "a"})}
    tables["b"]["columns"]["category"] = {"_stype": stype.categorical, "categories": [0, 1, 2]}
    skeleton = Skeleton.from_arrays({"a": 3, "b": 4}, {role: [0, 0, 1, -1]})
    generator = GlobalSCMGenerator(
        tables,
        ColumnGraph((source, target), (edge,), (source, target)),
        skeleton,
        tiny_config(scm_params=SCMParams(mlp_emb_dim=1)),
        0,
    )
    generator.encoders[edge.identity] = lambda indices: torch.tensor([0.0, 10.0, 1.0])[
        indices
    ].unsqueeze(1)
    generator.decoders[target] = lambda x: x[:, 0]
    generator.noise_distributions[target] = ZeroNoise()
    frames = generator.evaluate({source: np.array([0, 2, 1, 0])})
    np.testing.assert_array_equal(frames["a"]["score"], [0.5, 10.0, 0.0])


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("policy", ["bounded", "full"])
def test_real_generation_is_reproducible(seed, policy):
    config = tiny_config(rscm=RSCMParams(parent_policy=policy, batch_size=3))
    a, b = SyntheticDataset(seed, config), SyntheticDataset(seed, config)
    first, second = a.make_db(), b.make_db()
    assert a.column_graph.fingerprint() == b.column_graph.fingerprint()
    for name in first.table_dict:
        pd.testing.assert_frame_equal(first.table_dict[name].df, second.table_dict[name].df)
    original = a.generator.evaluate()
    a.generator.config = replace(config, rscm=replace(config.rscm, batch_size=7))
    repeated = a.generator.evaluate()
    for name in original:
        np.testing.assert_allclose(
            original[name].select_dtypes("number").to_numpy(dtype=float, na_value=np.nan),
            repeated[name].select_dtypes("number").to_numpy(dtype=float, na_value=np.nan),
            rtol=1e-5,
            atol=1e-6,
        )


def test_relbench_cache_and_metadata(tmp_path):
    config = tiny_config(cache_dir=str(tmp_path / "cache"))
    dataset = SyntheticDataset(0, config)
    db = dataset.get_db(False)
    loaded = SyntheticDataset(0, config).get_db(False)
    for name, table in db.table_dict.items():
        pd.testing.assert_frame_equal(table.df, loaded.table_dict[name].df)
        assert table.time_col is None
    sidecar = json.loads((tmp_path / "cache/causal.json").read_text())
    assert sidecar["base_revision"].startswith("2a273cf")
    assert set(sidecar["tables"]) == set(db.table_dict)
    assert sidecar["scm_settings"]["mlp_emb_dim"] == 32
    with np.load(tmp_path / "cache/skeleton.npz", allow_pickle=False) as arrays:
        for key in sidecar["skeleton_archive"]["row_ids"].values():
            assert arrays[key].dtype.kind == "i"
    with pytest.raises(ValueError, match="configuration differs"):
        SyntheticDataset(1, config).get_db(False)
    with pytest.raises(ValueError, match="legacy"):
        SyntheticDataset(0, replace(config, generation_mode="legacy_scm")).get_db(False)


def test_sql_schema_and_implicit_row_identity(schema_sql):
    dataset = SyntheticDataset(0, tiny_config(schema_file=str(schema_sql)))
    db = dataset.make_db()
    assert set(db.table_dict) == {"users", "orders", "products", "order_items"}
    assert db.table_dict["order_items"].pkey_col == "row_idx"
    assert db.table_dict["users"].df["status"].dropna().isin(["active", "inactive", "banned"]).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"max_hops": 3},
        {"column_parent_cap": -1},
        {"cell_parent_cap_per_edge": 0},
        {"row_pooling": "max"},
        {"parent_policy": "truncate"},
        {"batch_size": 0},
    ],
)
def test_invalid_settings_fail(settings):
    with pytest.raises(ValueError):
        RSCMParams(**settings)


def test_memory_budget_and_configuration_identity():
    with pytest.raises(MemoryError):
        Skeleton.from_arrays({"a": 100}, {}, index_memory_bytes=1)
    assert config_fingerprint(Config()) == config_fingerprint(Config())
    with pytest.raises(ValueError, match="generation_mode"):
        Config(generation_mode="unknown")


def test_legacy_sql_without_primary_key_has_no_phantom_column(schema_sql):
    config = replace(tiny_config(schema_file=str(schema_sql)), generation_mode="legacy_scm")
    db = SyntheticDataset(0, config).make_db()
    table = db.table_dict["order_items"]
    assert table.pkey_col is None
    assert all(isinstance(name, str) for name in table.df.columns)
