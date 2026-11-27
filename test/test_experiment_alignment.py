import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from relbench.base import Database
from torch_frame import stype

from relrel.config import Choices, Config, DAGParams, DatabaseParams
from relrel.dag import ErdosRenyi
from relrel.dataset import SyntheticDataset
from relrel.experiments import RunSpec, export, synthesize

SCM_FAMILIES = {
    "ErdosRenyi",
    "BarabasiAlbert",
    "RandomTree",
    "ReverseRandomTree",
    "Layered",
}


def assert_foreign_keys_valid(database):
    for table in database.table_dict.values():
        assert table.df[table.pkey_col].is_unique
        for column, parent_name in table.fkey_col_to_pkey_table.items():
            parent = database.table_dict[parent_name]
            assert table.df[column].notna().all()
            assert table.df[column].isin(parent.df[parent.pkey_col]).all()


def test_paper_graph_family_prior_is_preserved():
    config = Config()
    assert config.scm_params.scm_layout_choices.kind == "set"
    assert len(config.scm_params.scm_layout_choices.value) == 5
    assert set(config.scm_params.scm_layout_choices.value) == SCM_FAMILIES
    assert config.database_params.table_layout_choices.value == [
        "BarabasiAlbert",
        "ReverseRandomTree",
        "WattsStrogatz",
    ]


@pytest.mark.parametrize("seed", [0, 1, 10000])
def test_local_original_is_bit_accurate_to_plurel_v1(seed, tmp_path):
    """DAG=local, DECODING=original, FIXED_MLP=0, DATES=activity, EXPORT_CAT=plurel_bool
    must match the unmodified PluRel v1.0.0 (2a273cf) generator bit-for-bit."""
    spec = RunSpec(
        dag="local",
        decoding="original",
        fixed_mlp=False,
        dates="activity",
        export_cat="plurel_bool",
        tiny=True,
    )
    synthesize(tmp_path, "local-orig", spec, count=1, seed_start=seed, workers=1)
    generated = Database.load(tmp_path / f"synthetic/raw/local-orig/db-{seed:08d}/db")

    # Compare directly against SyntheticDataset(seed, legacy_scm) AND original 2a273cf if git history exists.
    legacy_cfg = Config(
        database_params=DatabaseParams(
            num_tables_choices=Choices("set", [3]),
            num_cols_choices=Choices("range", [4, 4]),
            num_rows_entity_table_choices=Choices("set", [12]),
            num_rows_activity_table_choices=Choices("set", [24]),
        )
    )
    legacy = SyntheticDataset(seed, legacy_cfg).make_db()
    assert set(generated.table_dict) == set(legacy.table_dict)
    for name in sorted(legacy.table_dict):
        expected, actual = legacy.table_dict[name], generated.table_dict[name]
        assert actual.pkey_col == expected.pkey_col
        assert actual.fkey_col_to_pkey_table == expected.fkey_col_to_pkey_table
        assert actual.time_col == expected.time_col
        pd.testing.assert_frame_equal(actual.df, expected.df, check_exact=True)

    repo = Path(__file__).resolve().parents[1]
    check = subprocess.run(
        ["git", "cat-file", "-e", "2a273cf^{commit}"], cwd=repo, capture_output=True
    )
    if check.returncode == 0:
        with tempfile.TemporaryDirectory() as orig_tmp:
            archive = subprocess.check_output(["git", "archive", "2a273cf", "plurel"], cwd=repo)
            subprocess.run(["tar", "-x", "-C", orig_tmp], input=archive, check=True)
            sys.path.insert(0, orig_tmp)
            try:
                from plurel.config import (
                    Choices as PChoices,
                )
                from plurel.config import (
                    Config as PConfig,
                )
                from plurel.config import (
                    DatabaseParams as PDatabaseParams,
                )
                from plurel.dataset import SyntheticDataset as PSyntheticDataset

                p_cfg = PConfig(
                    database_params=PDatabaseParams(
                        num_tables_choices=PChoices("set", [3]),
                        num_cols_choices=PChoices("range", [4, 4]),
                        num_rows_entity_table_choices=PChoices("set", [12]),
                        num_rows_activity_table_choices=PChoices("set", [24]),
                    )
                )
                p_db = PSyntheticDataset(seed, p_cfg).make_db()
                assert set(generated.table_dict) == set(p_db.table_dict)
                for name in sorted(p_db.table_dict):
                    expected, actual = p_db.table_dict[name], generated.table_dict[name]
                    assert actual.pkey_col == expected.pkey_col
                    assert actual.fkey_col_to_pkey_table == expected.fkey_col_to_pkey_table
                    assert actual.time_col == expected.time_col
                    pd.testing.assert_frame_equal(actual.df, expected.df, check_exact=True)
            finally:
                sys.path.remove(orig_tmp)
                for mod in [k for k in sys.modules if k == "plurel" or k.startswith("plurel.")]:
                    del sys.modules[mod]


@pytest.mark.parametrize("dag", ["local", "global"])
@pytest.mark.parametrize("decoding", ["original", "quantile"])
def test_local_and_global_with_both_decoders_and_export(dag, decoding, tmp_path):
    spec = RunSpec(dag=dag, decoding=decoding, tiny=True)
    name = f"{dag}-{decoding}"
    synthesize(tmp_path, name, spec, count=2, seed_start=10000, workers=1)
    export(tmp_path, name)

    raw_manifest = json.loads((tmp_path / f"synthetic/raw/{name}/corpus.json").read_text())
    rt_manifest = json.loads((tmp_path / f"synthetic/rt/{name}/corpus.json").read_text())
    assert raw_manifest["count"] == 2
    assert len(rt_manifest["databases"]) == 2
    assert rt_manifest["tasks"]

    for entry in raw_manifest["databases"]:
        db_id = entry["id"]
        db = Database.load(tmp_path / f"synthetic/rt/{name}/{db_id}/db")
        assert_foreign_keys_valid(db)
        assert any(table.time_col == "date" for table in db.table_dict.values())
        causal = json.loads((tmp_path / f"synthetic/raw/{name}/{db_id}/causal.json").read_text())
        if dag == "global":
            assert causal["column_graph"]["metadata"]["proposal_family"] in SCM_FAMILIES
            if decoding == "quantile":
                assert causal.get("categorical_calibration")
            else:
                assert not causal.get("categorical_calibration")
        else:
            assert all(t["family"] in SCM_FAMILIES for t in causal["local_scm"].values())
            if decoding == "quantile":
                assert any(t["quantiles"] for t in causal["local_scm"].values())
            else:
                assert not any(t["quantiles"] for t in causal["local_scm"].values())

    for db_id, table_name, target, _, kind in rt_manifest["tasks"]:
        db = Database.load(tmp_path / f"synthetic/rt/{name}/{db_id}/db")
        table = db.table_dict[table_name]
        assert target not in (table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table)
        assert table.df[target].nunique(dropna=True) > 1
        assert (kind == "clf") == pd.api.types.is_bool_dtype(table.df[target])


def test_direct_sampler_matches_family_edges_and_seeds(tmp_path):
    fam_spec = RunSpec(dag="global", decoding="quantile", sampler="family", tiny=True)
    dir_spec = RunSpec(dag="global", decoding="quantile", sampler="direct", tiny=True)
    synthesize(tmp_path, "fam", fam_spec, count=3, seed_start=10000, workers=1)
    synthesize(
        tmp_path,
        "dir",
        dir_spec,
        count=3,
        seed_start=10000,
        workers=1,
        match_edges_from="fam",
    )
    fam_m = json.loads((tmp_path / "synthetic/raw/fam/corpus.json").read_text())
    dir_m = json.loads((tmp_path / "synthetic/raw/dir/corpus.json").read_text())
    assert [d["seed"] for d in dir_m["databases"]] == [d["seed"] for d in fam_m["databases"]]
    assert [d["edges"] for d in dir_m["databases"]] == [d["edges"] for d in fam_m["databases"]]
    assert [d["skeleton_sha256"] for d in dir_m["databases"]] == [
        d["skeleton_sha256"] for d in fam_m["databases"]
    ]


def test_hyperparam_controls_hops_pooling_dates_and_export_cat(tmp_path):
    spec_h1 = RunSpec(
        dag="global",
        decoding="quantile",
        max_hops=1,
        parent_policy="full",
        row_pooling="sum",
        dates="none",
        export_cat="raw_int",
        tiny=True,
    )
    synthesize(tmp_path, "h1-static", spec_h1, count=1, seed_start=10000, workers=1)
    db = Database.load(tmp_path / "synthetic/raw/h1-static/db-00010000/db")
    assert all(table.time_col is None for table in db.table_dict.values())
    causal = json.loads((tmp_path / "synthetic/raw/h1-static/db-00010000/causal.json").read_text())
    assert all(len(edge["path"]) <= 1 for edge in causal["column_graph"]["edges"])
    assert causal["settings"]["parent_policy"] == "full"
    assert causal["settings"]["row_pooling"] == "sum"
    assert any(
        is_integer_dtype(table.df[col]) and table.df[col].nunique() > 2
        for table in db.table_dict.values()
        for col in table.df
        if col != table.pkey_col and col not in table.fkey_col_to_pkey_table
    )

    spec_text = RunSpec(
        dag="local",
        decoding="quantile",
        fixed_mlp=True,
        export_cat="text",
        tiny=True,
    )
    synthesize(tmp_path, "local-text", spec_text, count=1, seed_start=10000, workers=1)
    export(tmp_path, "local-text")
    db_text = Database.load(tmp_path / "synthetic/rt/local-text/db-00010000/db")
    assert any(
        pd.api.types.is_string_dtype(table.df[col])
        for table in db_text.table_dict.values()
        for col in table.df
        if col not in (table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table)
    )


def test_validation_rejects_invalid_hyperparam_combinations():
    with pytest.raises(ValueError, match="require --dag=global"):
        RunSpec(dag="local", sampler="direct").validate()
    with pytest.raises(ValueError, match="require --sampler=direct"):
        RunSpec(dag="global", sampler="family", degree=(2.0,)).validate()
    with pytest.raises(ValueError, match="requires --degree or --match-edges-from"):
        RunSpec(dag="global", sampler="direct").validate()


def test_decoder_swap_preserves_numeric_only_global_corpus(tmp_path):
    base = RunSpec(dag="global", decoding="original", tiny=True).to_config()
    num_only = replace(
        base,
        scm_params=replace(base.scm_params, col_stype_choices=Choices("set", [stype.numerical])),
    )
    quant = replace(num_only, rscm=replace(num_only.rscm, categorical_alpha_choices=(1.0, 5.0)))
    db_orig = SyntheticDataset(10000, num_only).make_db()
    db_quant = SyntheticDataset(10000, quant).make_db()
    for name, table in db_orig.table_dict.items():
        pd.testing.assert_frame_equal(table.df, db_quant.table_dict[name].df, check_exact=True)


def test_erdos_renyi_deadlock_guard_terminates():
    dag = ErdosRenyi(
        num_nodes=4,
        dag_params=DAGParams(er_p_choices=Choices("set", [0.15])),
        seed=0,
    )
    assert dag.size() == 4


def test_global_schema_rng_does_not_clobber_keyed_stream():
    config = RunSpec(dag="global", decoding="quantile", tiny=True).to_config()
    for seed in (0, 1):
        np.random.seed(999)
        first = SyntheticDataset(seed, config)._prepare_global_schema()
        np.random.seed(12345)
        second = SyntheticDataset(seed, config)._prepare_global_schema()
        assert list(first) == list(second)
        for name in first:
            assert first[name]["columns"] == second[name]["columns"]
            assert first[name]["fkey_col_to_pkey_table"] == second[name]["fkey_col_to_pkey_table"]


def test_global_non_fixed_mlp_is_deterministic_and_sidecar_accurate():
    spec = RunSpec(dag="global", decoding="quantile", fixed_mlp=False, dates="activity", tiny=True)
    config = spec.to_config()
    np.random.seed(111)
    ds1 = SyntheticDataset(10000, config)
    db1 = ds1.make_db()
    np.random.seed(99999)
    ds2 = SyntheticDataset(10000, config)
    db2 = ds2.make_db()
    for name in sorted(db1.table_dict):
        pd.testing.assert_frame_equal(db1.table_dict[name].df, db2.table_dict[name].df)
    assert "per-call hidden activations" in ds1.causal_metadata["mechanisms"]
    assert "activity tables dated" in ds1.causal_metadata["temporal_policy"]
