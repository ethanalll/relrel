"""Feature-independent rows, observed scalar FK links and inverse indexes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from relrel.bipartite import sample_bipartite_assignments
from relrel.config import Config
from relrel.rng import config_fingerprint, keyed_seed, scoped_rng
from relrel.rscm import FKRole
from relrel.utils import TableType

DEFAULT_INDEX_BYTES = 512 * 1024 * 1024


def check_memory(required: int, limit: int, purpose: str) -> None:
    if required > limit:
        raise MemoryError(
            f"{purpose} needs at least {required:,} bytes; configured limit is {limit:,}. "
            "Raise the explicit resource budget or reduce the corpus size."
        )


@dataclass(frozen=True)
class InverseIndex:
    """CSR child ordinals grouped by parent ordinal, with null links excluded."""

    offsets: np.ndarray
    child_rows: np.ndarray

    def group(self, parent_row: int) -> np.ndarray:
        return self.child_rows[self.offsets[parent_row] : self.offsets[parent_row + 1]]

    @property
    def nbytes(self) -> int:
        return self.offsets.nbytes + self.child_rows.nbytes


def inverse_index(
    assignments: np.ndarray, parent_count: int, child_row_ids: np.ndarray
) -> InverseIndex:
    valid_rows = np.flatnonzero(assignments >= 0)
    # Stable identities, rather than storage positions, define within-group order.
    order = np.lexsort((child_row_ids[valid_rows], assignments[valid_rows]))
    child_rows = valid_rows[order].astype(np.int64, copy=False)
    counts = np.bincount(assignments[valid_rows], minlength=parent_count)
    offsets = np.empty(parent_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    offsets.setflags(write=False)
    child_rows.setflags(write=False)
    return InverseIndex(offsets, child_rows)


def _scalar_assignments(values: Any, count: int, parent_count: int, role: FKRole) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FK {role.key} must have one scalar-or-null value per child row") from exc
    if raw.ndim != 1 or len(raw) != count:
        raise ValueError(f"FK {role.key} must have one scalar-or-null value per child row")
    if raw.dtype.kind in "iu":
        if np.any(raw >= parent_count) or (raw.dtype.kind == "i" and np.any(raw < -1)):
            raise ValueError(f"FK {role.key} contains an out-of-range parent row ordinal")
        result = raw.astype(np.int64, copy=True)
    elif raw.dtype.kind in "fO":
        result = np.empty(count, dtype=np.int64)
        for index, value in enumerate(raw):
            if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                result[index] = -1
            elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                if value < -1 or value >= parent_count:
                    raise ValueError(f"FK {role.key} contains an out-of-range parent row ordinal")
                result[index] = int(value)
            elif (
                isinstance(value, (float, np.floating))
                and np.isfinite(value)
                and value.is_integer()
            ):
                if value < -1 or value >= parent_count:
                    raise ValueError(f"FK {role.key} contains an out-of-range parent row ordinal")
                result[index] = int(value)
            else:
                raise ValueError(
                    f"FK {role.key} contains a non-scalar integer reference: {value!r}"
                )
    else:
        raise ValueError(f"FK {role.key} must contain scalar integer row ordinals or nulls")
    if np.any((result < -1) | (result >= parent_count)):
        raise ValueError(f"FK {role.key} contains an out-of-range parent row ordinal")
    result.setflags(write=False)
    return result


@dataclass
class Skeleton:
    """Rows and FK observations that can be constructed before feature values.

    ``-1`` is the null sentinel. Multiple child rows may reference one parent;
    each individual child FK is scalar. Stable row identities permit transport
    of frozen state when row storage is permuted.
    """

    row_counts: dict[str, int]
    fk_assignments: dict[FKRole, np.ndarray]
    row_ids: dict[str, np.ndarray]
    inverse_indexes: dict[FKRole, InverseIndex]
    metadata: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    schema_tables: dict[str, dict] = field(default_factory=dict)
    index_memory_bytes: int = DEFAULT_INDEX_BYTES
    counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        row_counts: dict[str, int],
        fk_assignments: dict[FKRole, Any],
        row_ids: dict[str, Any] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
        index_memory_bytes: int = DEFAULT_INDEX_BYTES,
        schema_tables: dict[str, dict] | None = None,
    ) -> Skeleton:
        counts = dict(row_counts)
        for name, count in counts.items():
            if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
                raise ValueError(f"Invalid row count for {name}: {count!r}")
            counts[name] = int(count)
        if index_memory_bytes <= 0:
            raise ValueError("index_memory_bytes must be positive")
        # Check the upper bound before constructing assignment/index arrays.
        required = sum(counts.values()) * 8
        role_identities = set()
        for role in fk_assignments:
            if not isinstance(role, FKRole):
                raise TypeError("FK assignment keys must be named FKRole records")
            if role.identity in role_identities:
                raise ValueError(f"Duplicate named FK role: {role.key}")
            role_identities.add(role.identity)
            if role.child_table == role.parent_table:
                raise ValueError("Self-referential foreign keys are unsupported")
            if role.child_table not in counts or role.parent_table not in counts:
                raise ValueError(f"FK {role.key} references an unknown table")
            required += (counts[role.child_table] * 2 + counts[role.parent_table] + 1) * 8
        for table, fields in (metadata or {}).items():
            if table not in counts:
                raise ValueError(f"Metadata references unknown table {table}")
            for values in fields.values():
                required += np.asarray(values).nbytes
        check_memory(required, index_memory_bytes, "Relational skeleton indexes")
        identities = {}
        for table, count in counts.items():
            values = np.asarray(
                row_ids[table] if row_ids and table in row_ids else np.arange(count)
            )
            if values.ndim != 1 or len(values) != count or values.dtype.kind not in "iu":
                raise ValueError(
                    f"Stable identities for {table} must be a one-dimensional integer array"
                )
            values = values.astype(np.int64, copy=True)
            if len(np.unique(values)) != count:
                raise ValueError(f"Stable row identities for {table} must be unique")
            values.setflags(write=False)
            identities[table] = values
        links, inverses = {}, {}
        for role, values in sorted(fk_assignments.items(), key=lambda item: item[0].identity):
            links[role] = _scalar_assignments(
                values, counts[role.child_table], counts[role.parent_table], role
            )
            inverses[role] = inverse_index(
                links[role], counts[role.parent_table], identities[role.child_table]
            )
        structural = {}
        for table, fields in (metadata or {}).items():
            if table not in counts:
                raise ValueError(f"Metadata references unknown table {table}")
            structural[table] = {}
            for name, values in fields.items():
                array = np.asarray(values)
                if (
                    array.ndim != 1
                    or len(array) != counts[table]
                    or array.dtype.kind not in "biufU"
                    or (array.dtype.kind == "f" and array.dtype.itemsize > 8)
                ):
                    raise ValueError(
                        f"Structural metadata {table}.{name} must be a scalar row array"
                    )
                if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"Structural metadata {table}.{name} must be finite; use -1 for null metadata"
                    )
                structural[table][name] = array.copy()
                structural[table][name].setflags(write=False)
        result = cls(
            counts, links, identities, inverses, structural, schema_tables or {}, index_memory_bytes
        )
        check_memory(result.nbytes, index_memory_bytes, "Relational skeleton indexes")
        result.counters = {"index_rows": sum(len(array) for array in links.values())}
        return result

    @classmethod
    def from_schema(cls, table_dict: dict[str, dict], config: Config, seed: int = 0) -> Skeleton:
        """Reuse row-count and hierarchical-SBM priors using independent FK streams.

        Rows in the same child leaf block reuse one parent probability vector.
        References are sampled before any feature values are generated.
        """
        limits = config.rscm
        counts = {}
        for table, info in sorted(table_dict.items()):
            if "num_rows" in info:
                value = info["num_rows"]
                if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Invalid row count for {table}: {value!r}")
                counts[table] = int(value)
            else:
                table_type = info.get("type", TableType.Entity)
                choices = (
                    config.database_params.num_rows_activity_table_choices
                    if table_type == TableType.Activity
                    else config.database_params.num_rows_entity_table_choices
                )
                with scoped_rng(keyed_seed(seed, "row-count", table)):
                    counts[table] = int(choices.sample_uniform())
        links = {}
        scm = config.scm_params
        # Preflight the complete retained skeleton before allocating/sampling
        # any role. A tiny index budget must not first build enormous FK arrays.
        required = sum(counts.values()) * 8
        for child, info in table_dict.items():
            for name, parent in info.get("fkey_col_to_pkey_table", {}).items():
                FKRole(name, child, parent)
                if parent not in counts:
                    raise ValueError(f"FK {child}.{name} references unknown table {parent}")
                required += (counts[child] * 2 + counts[parent] + 1) * 8
        check_memory(required, limits.index_memory_bytes, "Relational skeleton indexes")
        for child, info in sorted(table_dict.items()):
            for name, parent in sorted(info.get("fkey_col_to_pkey_table", {}).items()):
                role = FKRole(name, child, parent)
                if parent not in counts:
                    raise ValueError(f"FK {role.key} references unknown table {parent}")
                size_a, size_b = counts[parent], counts[child]
                if size_a == 0 or size_b == 0:
                    assignments = np.full(size_b, -1, dtype=np.int64)
                else:
                    # Reserve parent-sized affinity, indexing and CDF buffers.
                    check_memory(
                        size_a * 8 * 4, limits.temp_memory_bytes, "FK assignment workspace"
                    )
                    with scoped_rng(keyed_seed(seed, "fk-assignments", role.identity)):
                        levels = scm.bi_hsbm_levels_choices.sample_uniform()
                        hierarchy_a = [
                            scm.bi_hsbm_clusters_per_level_choices.sample_uniform()
                            for _ in range(levels)
                        ]
                        hierarchy_b = [
                            scm.bi_hsbm_clusters_per_level_choices.sample_uniform()
                            for _ in range(levels)
                        ]
                        fixed_workspace = (
                            (size_a + size_b) * levels * 8
                            + size_a * 32
                            + size_b * 16
                            + sum(a * b for a, b in zip(hierarchy_a, hierarchy_b)) * 16
                        )
                        check_memory(
                            fixed_workspace + size_a * 8 * 4,
                            limits.temp_memory_bytes,
                            "FK assignment workspace",
                        )
                        assignments = sample_bipartite_assignments(
                            size_a,
                            size_b,
                            hierarchy_a,
                            hierarchy_b,
                            chunk_memory_bytes=limits.temp_memory_bytes - fixed_workspace,
                        )
                links[role] = assignments
        return cls.from_arrays(
            counts, links, index_memory_bytes=limits.index_memory_bytes, schema_tables=table_dict
        )

    sample = from_schema

    @property
    def roles(self) -> tuple[FKRole, ...]:
        return tuple(sorted(self.fk_assignments, key=lambda role: role.identity))

    @property
    def nbytes(self) -> int:
        return (
            sum(array.nbytes for array in self.row_ids.values())
            + sum(array.nbytes for array in self.fk_assignments.values())
            + sum(index.nbytes for index in self.inverse_indexes.values())
            + sum(array.nbytes for fields in self.metadata.values() for array in fields.values())
        )

    def forward(self, role: FKRole) -> np.ndarray:
        return self.fk_assignments[role]

    def inverse(self, role: FKRole) -> InverseIndex:
        return self.inverse_indexes[role]

    def structural_field(self, table: str, name: str) -> np.ndarray:
        if name == "row_id" or name == self.schema_tables.get(table, {}).get("pkey_col"):
            return self.row_ids[table]
        if name in self.metadata.get(table, {}):
            return self.metadata[table][name]
        for role, values in self.fk_assignments.items():
            if role.child_table == table and role.name == name:
                return values
        raise ValueError(
            f"{table}.{name} is not fixed structural metadata; feature-dependent filters are unsupported"
        )

    def to_dict(self) -> dict:
        state = {
            "row_counts": dict(sorted(self.row_counts.items())),
            "row_ids": {table: values.tolist() for table, values in sorted(self.row_ids.items())},
            "foreign_keys": [
                {
                    "name": role.name,
                    "child_table": role.child_table,
                    "parent_table": role.parent_table,
                    "assignments": self.fk_assignments[role].tolist(),
                }
                for role in self.roles
            ],
            "metadata": {
                table: {name: values.tolist() for name, values in sorted(fields.items())}
                for table, fields in sorted(self.metadata.items())
            },
        }
        # Float32 comparisons follow float32 scalar promotion. Restoring plain
        # JSON lists as float64 can therefore change a structural predicate.
        # Empty-metadata snapshots keep their existing structural fingerprint.
        if any(self.metadata.values()):
            state["metadata_dtypes"] = {
                table: {name: values.dtype.str for name, values in sorted(fields.items())}
                for table, fields in sorted(self.metadata.items())
            }
        return state

    @classmethod
    def from_dict(cls, state: dict, **kwargs) -> Skeleton:
        roles = {
            FKRole(item["name"], item["child_table"], item["parent_table"]): item["assignments"]
            for item in state["foreign_keys"]
        }
        metadata = state.get("metadata", {})
        if "metadata_dtypes" in state:
            dtypes = state["metadata_dtypes"]
            metadata = {
                table: {
                    name: np.asarray(values, dtype=dtypes[table][name])
                    for name, values in fields.items()
                }
                for table, fields in metadata.items()
            }
        return cls.from_arrays(
            state["row_counts"],
            roles,
            row_ids=state.get("row_ids"),
            metadata=metadata,
            **kwargs,
        )

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self.to_dict())
