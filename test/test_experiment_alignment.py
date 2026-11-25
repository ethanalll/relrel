import json
from dataclasses import replace

import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from relbench.base import Database
from torch_frame import stype

from relrel import experiments
from relrel.config import Choices, Config
from relrel.dataset import SyntheticDataset
from relrel.experiments import experiment_config, export, generate_reference, make_case, write_json

SCM_FAMILIES = {
    "ErdosRenyi",
    "BarabasiAlbert",
    "RandomTree",
    "ReverseRandomTree",
    "Layered",
}
ARMS = {
    "local-original": (False, False),
    "local-quantile": (False, True),
    "global-original": (True, False),
    "global-quantile": (True, True),
}


def assert_foreign_keys_valid(database):
    for table in database.table_dict.values():
        assert table.df[table.pkey_col].is_unique
        for column, parent_name in table.fkey_col_to_pkey_table.items():
            parent = database.table_dict[parent_name]
            assert table.df[column].notna().all()
            assert table.df[column].isin(parent.df[parent.pkey_col]).all()


def assert_paired_structure(reference, arm):
    assert reference.table_dict.keys() == arm.table_dict.keys()
    for name, expected in reference.table_dict.items():
        actual = arm.table_dict[name]
        assert actual.pkey_col == expected.pkey_col
        assert actual.fkey_col_to_pkey_table == expected.fkey_col_to_pkey_table
        assert actual.time_col == expected.time_col
        keys = [expected.pkey_col, *expected.fkey_col_to_pkey_table]
        if expected.time_col:
            keys.append(expected.time_col)
        pd.testing.assert_frame_equal(actual.df[keys], expected.df[keys])
        pd.testing.assert_series_equal(actual.df.dtypes, expected.df.dtypes)
        pd.testing.assert_frame_equal(actual.df.isna(), expected.df.isna())


def test_paper_graph_family_prior_is_preserved():
    config = Config()
    assert config.scm_params.scm_layout_choices.kind == "set"
    assert len(config.scm_params.scm_layout_choices.value) == 5
    assert set(config.scm_params.scm_layout_choices.value) == SCM_FAMILIES
    assert experiment_config(False).scm_params.scm_layout_choices == (
        config.scm_params.scm_layout_choices
    )
    assert config.database_params.table_layout_choices.value == [
        "BarabasiAlbert",
        "ReverseRandomTree",
        "WattsStrogatz",
    ]


def test_canonical_generators_have_different_timestamp_contracts():
    config = experiment_config(True)
    legacy = SyntheticDataset(10000, replace(config, generation_mode="legacy_scm")).make_db()
    global_config = replace(
        config,
        generation_mode="global_rscm",
        rscm=replace(config.rscm, categorical_alpha_choices=(1.0, 5.0)),
    )
    global_data = SyntheticDataset(10000, global_config).make_db()
    assert_foreign_keys_valid(legacy)
    assert_foreign_keys_valid(global_data)
    dates = [table.df["date"] for table in legacy.table_dict.values() if table.time_col == "date"]
    assert dates
    assert all(series.is_monotonic_increasing and series.is_unique for series in dates)
    assert all(table.time_col is None for table in global_data.table_dict.values())
    assert all("date" not in table.df for table in global_data.table_dict.values())


@pytest.mark.parametrize("reference", ["plurel", "relrel"])
def test_reference_exports_keep_their_own_table_contract(reference, tmp_path):
    generate_reference(tmp_path, reference, count=1, seed=10000, tiny=True)
    output = tmp_path / "synthetic/reference" / reference
    manifest = json.loads((output / "corpus.json").read_text())
    assert manifest["reference_revision"] == experiments.REFERENCE_REVISIONS[reference]
    assert manifest["databases"] == [{"id": "db-00010000", "seed": 10000}]
    database = Database.load(output / "db-00010000/db")
    assert_foreign_keys_valid(database)
    if reference == "plurel":
        assert any(table.time_col == "date" for table in database.table_dict.values())
    else:
        assert all(table.time_col is None for table in database.table_dict.values())
        generation = json.loads((output / "db-00010000/generation.json").read_text())
        assert generation["categorical_export_version"] == 2
        assert any(
            is_integer_dtype(table.df[column]) and table.df[column].nunique() > 2
            for table in database.table_dict.values()
            for column in table.df
            if column != table.pkey_col and column not in table.fkey_col_to_pkey_table
        )


@pytest.mark.parametrize("seed", [10000, 10001, 10002])
def test_factorial_arms_control_schema_and_graphs(seed, tmp_path):
    case = make_case(seed, "factorial", True, tmp_path)
    assert "rejected" not in case, case
    assert case["arms"].keys() == ARMS.keys()
    generated = {}
    for arm, (global_dag, quantiles) in ARMS.items():
        info = case["arms"][arm]
        assert info["global_dag"] is global_dag
        assert info["quantiles"] is quantiles
        assert info["tasks"] == case["arms"]["local-original"]["tasks"]
        assert info["tasks"]
        path = tmp_path / arm / case["id"]
        db = Database.load(path / "db")
        assert_foreign_keys_valid(db)
        for _, table_name, target, _, _ in info["tasks"]:
            table = db.table_dict[table_name]
            assert target not in (table.time_col, table.pkey_col, *table.fkey_col_to_pkey_table)
            assert table.df[target].nunique(dropna=True) > 1
        generated[arm] = (db, json.loads((path / "causal.json").read_text()))

    reference, local = generated["local-original"]
    assert any(table.time_col == "date" for table in reference.table_dict.values())
    for db, _ in generated.values():
        assert_paired_structure(reference, db)
    local_quantile = generated["local-quantile"][1]
    for name, diagnostics in local["local_scm"].items():
        assert diagnostics["family"] in SCM_FAMILIES
        assert diagnostics["graph_sha256"] == local_quantile["local_scm"][name]["graph_sha256"]
        assert not diagnostics["quantiles"]
    assert any(table["quantiles"] for table in local_quantile["local_scm"].values())

    global_original = generated["global-original"][1]
    global_quantile = generated["global-quantile"][1]
    assert global_original["column_graph"] == global_quantile["column_graph"]
    assert global_original["column_graph"]["metadata"]["proposal_family"] in SCM_FAMILIES
    assert not global_original.get("categorical_calibration")
    assert global_quantile["categorical_calibration"]


def test_factorial_replay_and_export_keep_dates_and_targets(tmp_path):
    seed = 10000
    source = tmp_path / "synthetic/raw/factorial/train"
    source.mkdir(parents=True)
    repeated = tmp_path / "repeated"
    repeated.mkdir()
    case = make_case(seed, "factorial", True, source)
    assert "rejected" not in case, case
    assert make_case(seed, "factorial", True, repeated) == case
    write_json(
        source / "corpus.json",
        {"cases": [case], "generation_profile": experiments.GENERATION_PROFILE},
    )
    export(tmp_path, "factorial", "train")

    output = tmp_path / "synthetic/rt/factorial/train"
    for arm in ARMS:
        database = Database.load(source / arm / case["id"] / "db")
        replay = Database.load(repeated / arm / case["id"] / "db")
        exported = Database.load(output / arm / case["id"] / "db")
        for name, table in database.table_dict.items():
            pd.testing.assert_frame_equal(table.df, replay.table_dict[name].df)
            assert exported.table_dict[name].time_col == table.time_col
            if table.time_col:
                pd.testing.assert_series_equal(
                    exported.table_dict[name].df[table.time_col], table.df[table.time_col]
                )
        manifest = json.loads((output / arm / "corpus.json").read_text())
        assert manifest["generation_profile"] == experiments.GENERATION_PROFILE
        assert manifest["tasks"] == case["arms"][arm]["tasks"]
        assert manifest["databases"][0]["seed"] == seed


@pytest.mark.parametrize("family", sorted(SCM_FAMILIES))
def test_each_paper_family_works_in_both_dag_scopes(family, tmp_path, monkeypatch):
    config = experiment_config(True)
    config = replace(
        config,
        scm_params=replace(config.scm_params, scm_layout_choices=Choices("set", [family])),
    )
    monkeypatch.setattr(experiments, "experiment_config", lambda tiny: config)
    case = make_case(10000, "factorial", True, tmp_path)
    assert "rejected" not in case, case

    for arm, (global_dag, _) in ARMS.items():
        causal = json.loads((tmp_path / arm / case["id"] / "causal.json").read_text())
        if global_dag:
            assert causal["column_graph"]["metadata"]["proposal_family"] == family
        else:
            assert all(table["family"] == family for table in causal["local_scm"].values())


def test_decoder_swap_preserves_numeric_only_corpora(tmp_path, monkeypatch):
    config = experiment_config(True)
    config = replace(
        config,
        scm_params=replace(config.scm_params, col_stype_choices=Choices("set", [stype.numerical])),
    )
    monkeypatch.setattr(experiments, "experiment_config", lambda tiny: config)
    case = make_case(10000, "factorial", True, tmp_path)
    assert "rejected" not in case, case
    for scope in ("local", "global"):
        original = Database.load(tmp_path / f"{scope}-original" / case["id"] / "db")
        quantile = Database.load(tmp_path / f"{scope}-quantile" / case["id"] / "db")
        for name, table in original.table_dict.items():
            pd.testing.assert_frame_equal(table.df, quantile.table_dict[name].df, check_exact=True)


@pytest.mark.parametrize("seed", [10000, 10001, 10002, 10003])
def test_holdout_preserves_paired_scope_and_decoder_selection(seed, tmp_path):
    case = make_case(seed, "holdout", True, tmp_path)
    assert "rejected" not in case, case
    assert list(case["arms"]) == ["reference"]
    reference = case["arms"]["reference"]
    assert reference["global_dag"] is bool(seed % 2)
    assert reference["quantiles"] is bool((seed // 2) % 2)
    db = Database.load(tmp_path / "reference" / case["id"] / "db")
    assert any(table.time_col == "date" for table in db.table_dict.values())
