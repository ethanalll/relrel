"""Generate RSCM databases using the v1.0.0 PluRel runtime and storage format."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relrel import Choices, Config, DatabaseParams, RSCMParams, SyntheticDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-hops", type=int, choices=[1, 2], default=2)
    parser.add_argument("--column-parent-cap", type=int, default=8)
    parser.add_argument("--parent-policy", choices=["bounded", "full"], default="bounded")
    parser.add_argument("--cell-parent-cap", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--schema-file", type=Path)
    parser.add_argument("--tiny", action="store_true", help="3 tables, 3 features, 8/12 rows")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    try:
        rscm = RSCMParams(
            max_hops=args.max_hops,
            column_parent_cap=args.column_parent_cap,
            parent_policy=args.parent_policy,
            cell_parent_cap_per_edge=args.cell_parent_cap,
            row_pooling=args.pooling,
            batch_size=args.batch_size,
        )
    except ValueError as error:
        parser.error(str(error))
    database = DatabaseParams()
    if args.tiny:
        database = DatabaseParams(
            num_tables_choices=Choices("set", [3]),
            num_cols_choices=Choices("range", [3, 3]),
            num_rows_entity_table_choices=Choices("set", [8]),
            num_rows_activity_table_choices=Choices("set", [12]),
        )
    for seed in range(args.seed_offset, args.seed_offset + args.count):
        output = args.output_dir.expanduser().resolve() / f"global-rscm-{seed}"
        print(f"START seed={seed} output={output}", flush=True)
        dataset = SyntheticDataset(
            seed,
            Config(
                generation_mode="global_rscm",
                rscm=rscm,
                database_params=database,
                schema_file=str(args.schema_file) if args.schema_file else None,
                cache_dir=str(output),
            ),
        )
        db = dataset.get_db(upto_test_timestamp=False)
        graph = dataset.causal_metadata["column_graph"]
        print(
            json.dumps(
                {
                    "seed": seed,
                    "output": str(output),
                    "tables": {name: len(table) for name, table in db.table_dict.items()},
                    "feature_columns": len(graph["columns"]),
                    "column_edges": len(graph["edges"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
