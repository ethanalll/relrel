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
    marker = tmp_path / "cache/generation.json"
    old_identity = json.loads(marker.read_text())
    del old_identity["categorical_export_version"]
    marker.write_text(json.dumps(old_identity))
    with pytest.raises(ValueError, match="older categorical export"):
        SyntheticDataset(0, config).get_db(False)


@pytest.mark.parametrize("alphas", [None, (1, 5)])
def test_sql_schema_and_implicit_row_identity(schema_sql, alphas):
    config = tiny_config(
        schema_file=str(schema_sql), rscm=RSCMParams(categorical_alpha_choices=alphas)
    )
    dataset = SyntheticDataset(0, config)
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
        {"categorical_alpha_choices": (0,)},
        {"column_degree_choices": ()},
        {"column_degree_choices": (-1,)},
    ],
)
def test_invalid_settings_fail(settings):
    with pytest.raises(ValueError):
        RSCMParams(**settings)


def test_memory_budget_and_configuration_identity():
    with pytest.raises(MemoryError):
        Skeleton.from_arrays({"a": 100}, {}, index_memory_bytes=1)
    assert config_fingerprint(Config()) == config_fingerprint(Config())
    assert Config().generation_mode == "global_rscm"
    with pytest.raises(ValueError, match="generation_mode"):
        Config(generation_mode="legacy_scm")


def test_categorical_export_keeps_constant_features_and_multiclass_ids():
    frame = pd.DataFrame(
        {
            "binary": [1, 1, 1],
            "constant_multiclass": [2, 2, 2],
            "multiclass": [0, 1, 2],
            "string": [1, 1, 1],
            "numeric": [4.0, 4.0, 4.0],
        }
    )
    columns = {
        name: {"_stype": stype.categorical, "categories": categories}
        for name, categories in {
            "binary": [0, 1],
            "constant_multiclass": [0, 1, 2],
            "multiclass": [0, 1, 2],
            "string": ["a", "b"],
        }.items()
    }
    columns["numeric"] = {"_stype": stype.numerical, "categories": None}
    exported = SyntheticDataset.process_categorical_data(frame, columns)
    pd.testing.assert_frame_equal(
        exported,
        pd.DataFrame(
            {
                "binary": [True, True, True],
                "constant_multiclass": [2, 2, 2],
                "multiclass": [0, 1, 2],
                "string": ["b", "b", "b"],
                "numeric": [4.0, 4.0, 4.0],
            }
        ),
    )


def test_column_degree_prior_respects_budget_locality_and_cap():
    columns = [Column(table, f"x{i}") for table in "abc" for i in range(6)]
    roles = [FKRole("a_id", "b", "a"), FKRole("b_id", "c", "b")]
    params = RSCMParams(max_hops=1, column_degree_choices=(0, 2, 4), column_parent_cap=3)
    draws = set()
    for seed in range(12):
        graph = sample_column_graph(columns, roles, SCMParams(), Config().dag_params, params, seed)
        target = graph.metadata["target_degree"]
        draws.add(target)
        assert 2 * len(graph.edges) / len(columns) == target
        assert all(len(edge.path) <= 1 for edge in graph.edges)
        assert all(len(graph.incoming(column)) <= 3 for column in columns)
    assert draws == {0, 2, 4}

    params = replace(params, column_degree_choices=(20,))
    graph = sample_column_graph(columns, roles, SCMParams(), Config().dag_params, params)
    assert len(graph.edges) == graph.metadata["eligible_edges_after_cap"]
    assert graph.metadata["realized_degree"] < graph.metadata["target_degree"]


def test_quantile_cutoffs_match_prior_and_stay_fixed_under_intervention():
    source, target = Column("a", "source"), Column("a", "target")
    edge = EdgeSpec(source, target, ())
    count = 100
    tables = {"a": numeric_table(["source", "target"])}
    tables["a"]["columns"]["target"] = {"_stype": stype.categorical, "categories": [0, 1, 2]}
    config = tiny_config(
        rscm=RSCMParams(categorical_alpha_choices=(5,), batch_size=13),
        scm_params=SCMParams(mlp_emb_dim=1),
    )
    generator = GlobalSCMGenerator(
        tables,
        ColumnGraph((source, target), (edge,), (source, target)),
        Skeleton.from_arrays({"a": count}, {}),
        config,
    )
    generator._source = lambda column: torch.arange(count, dtype=torch.float32)
    generator.encoders[edge.identity] = lambda values: values
    generator.decoders[target] = lambda latent: latent[:, 0]
    generator.noise_distributions[target] = ZeroNoise()
    # An intervention on the first call must still use observational cutoffs.
    changed = generator.evaluate({source: np.arange(count) + count})["a"]["target"]
    assert (changed == 2).all()
    calibration = generator.category_calibrations[target]
    baseline = generator.evaluate()["a"]["target"]
    cumulative = np.cumsum(calibration["baseline_counts"])[:-1]
    np.testing.assert_array_equal(
        cumulative, np.rint(count * np.cumsum(calibration["target_proportions"])[:-1])
    )
    np.testing.assert_array_equal(np.bincount(baseline), calibration["baseline_counts"])
    assert generator.category_calibrations[target] is calibration

    tied = Column("a", "tied")
    tables["a"]["columns"]["tied"] = {"_stype": stype.categorical, "categories": [0, 1]}
    tied_labels = generator._quantile_categories(tied, torch.zeros(count))
    assert len(tied_labels.unique()) == 1
    assert generator.category_calibrations[tied]["distinct_baseline_scores"] == 1


@pytest.mark.parametrize("category_count", [2, 3])
@pytest.mark.parametrize("alphas", [None, (1, 5)])
def test_export_preserves_columns_and_round_trips(tmp_path, category_count, alphas):
    config = tiny_config(
        cache_dir=str(tmp_path / "cache"),
        rscm=RSCMParams(categorical_alpha_choices=alphas, column_degree_choices=(2,)),
        scm_params=SCMParams(
            col_stype_choices=Choices("set", [stype.categorical]),
            num_categories_choices=Choices("set", [category_count]),
        ),
    )
    dataset = SyntheticDataset(3, config)
    exported = dataset.get_db(False)
    loaded = SyntheticDataset(3, config).get_db(False)
    for name, table in exported.table_dict.items():
        pd.testing.assert_frame_equal(table.df, loaded.table_dict[name].df)
    expected_calibrations = len(dataset.column_graph.columns) if alphas else 0
    assert len(dataset.generator.category_calibrations) == expected_calibrations
    for column, raw in dataset.generator.raw_values.items():
        values = exported.table_dict[column.table].df[column.name]
        assert values.dtype.kind == ("b" if category_count == 2 else "i")
        np.testing.assert_array_equal(values, raw.numpy())
