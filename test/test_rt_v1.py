import json
from dataclasses import replace

import networkx as nx
import pandas as pd
import pytest
from relbench.base import Database, Table

from relrel import Choices, Config, DatabaseParams
from relrel.rt_v1 import export_rt_v1
from relrel.schema import RandomSchemaGraphBuilder


def test_export_preserves_context_and_lists_only_trainable_targets(tmp_path):
    frame = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "numeric": [1.0, 2.0, 3.0],
            "constant_numeric": [4.0, 4.0, 4.0],
            "binary": [False, True, False],
            "constant_binary": [True, True, True],
            "multiclass": [0, 1, 2],
            "status": ["low", "middle", "high"],
        }
    )
    features = {
        "numeric": {"type": "numerical", "categories": None},
        "constant_numeric": {"type": "numerical", "categories": None},
        "binary": {"type": "categorical", "categories": [0, 1]},
        "constant_binary": {"type": "categorical", "categories": [0, 1]},
        "multiclass": {"type": "categorical", "categories": [0, 1, 2]},
        "status": {"type": "categorical", "categories": ["low", "middle", "high"]},
    }
    source, destination = tmp_path / "native", tmp_path / "rt" / "global-rscm-0"
    Database({"example": Table(frame, {}, pkey_col="id")}).save(source / "db")
    (source / "causal.json").write_text(json.dumps({"tables": {"example": {"features": features}}}))

    tasks = export_rt_v1(source, destination)
    output = Database.load(destination / "db").table_dict["example"].df
    assert set(output) == set(frame)
    assert tasks == [
        ("global-rscm-0", "example", "numeric", []),
        ("global-rscm-0", "example", "binary", []),
    ]
    assert json.loads((destination / "train_tasks.json").read_text()) == [
        list(task) for task in tasks
    ]
    assert output["binary"].tolist() == [False, True, False]
    assert output["constant_binary"].nunique() == 1
    assert isinstance(output["constant_binary"].iloc[0], str)
    assert output["multiclass"].nunique() == 3
    assert isinstance(output["multiclass"].iloc[0], str)
    assert output["status"].tolist() == frame["status"].tolist()
    pd.testing.assert_frame_equal(Database.load(source / "db").table_dict["example"].df, frame)


def test_random_schema_foreign_key_cap(monkeypatch):
    from relrel import schema

    graph = nx.DiGraph((parent, 6) for parent in range(6))
    monkeypatch.setitem(
        schema.DAG_REGISTRY, "fixture", lambda **_: type("DAG", (), {"graph": graph.copy()})()
    )
    config = Config(
        database_params=DatabaseParams(table_layout_choices=Choices("set", ["fixture"])),
    )
    uncapped = RandomSchemaGraphBuilder(config, 7, 0).build_graph()
    capped = RandomSchemaGraphBuilder(
        replace(config, rscm=replace(config.rscm, max_foreign_keys=5)), 7, 0
    ).build_graph()
    assert len(uncapped.nodes[6]["fkey_col_to_pkey_table"]) == 6
    assert len(capped.nodes[6]["fkey_col_to_pkey_table"]) == 5
    assert set(capped.edges).issubset(uncapped.edges)


def test_rt_export_rejects_schema_with_six_foreign_keys(tmp_path):
    source, destination = tmp_path / "native", tmp_path / "rt"
    columns = {f"fk{i}": [0] for i in range(6)}
    table = Table(pd.DataFrame({"id": [0], **columns}), dict.fromkeys(columns, "example"), "id")
    Database({"example": table}).save(source / "db")
    (source / "causal.json").write_text(json.dumps({"tables": {"example": {"features": {}}}}))
    with pytest.raises(ValueError, match="at most five foreign keys"):
        export_rt_v1(source, destination)
    assert not destination.exists()
