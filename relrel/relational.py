"""Indexed unique-endpoint grounding, exact pooling and direct parent sampling.

Paths start at the destination and end at the source. Every implementation
below builds reusable neighborhoods instead of scanning source/destination
row pairs. The forward/reverse shared-pivot case is especially important:
one source group can serve thousands of destination rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from relrel.rng import config_fingerprint, keyed_seed
from relrel.rscm import EdgeSpec
from relrel.skeleton import DEFAULT_INDEX_BYTES, Skeleton, check_memory, inverse_index

DEFAULT_TEMP_BYTES = 256 * 1024 * 1024


def sample_positions(population: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Floyd's uniform subset sampler: O(count) expected work, no full shuffle."""
    if population < 0 or count < 0 or count > population:
        raise ValueError("Subset size must be between zero and the population size")
    selected = set()
    positions = np.empty(count, dtype=np.int64)
    for index, end in enumerate(range(population - count, population)):
        position = int(rng.integers(0, end + 1))
        if position in selected:
            position = end
        selected.add(position)
        positions[index] = position
    return positions


def _filter_mask(skeleton: Skeleton, table: str, filters: Any) -> np.ndarray:
    mask = np.ones(skeleton.row_counts[table], dtype=bool)
    if not isinstance(filters, (list, tuple)):
        raise ValueError("Metadata filters must be a list of scalar field comparisons")
    for condition in filters:
        if not isinstance(condition, Mapping) or set(condition) - {"field", "op", "value"}:
            raise ValueError(
                "Unsupported metadata filter; pair-specific predicates are not supported"
            )
        if "field" not in condition or "op" not in condition:
            raise ValueError("Metadata filters need a field and an operation")
        values = skeleton.structural_field(table, condition["field"])
        operation = condition["op"]
        value = condition.get("value")
        if operation == "not_null":
            comparison = values != -1
            if values.dtype.kind == "f":
                comparison &= np.isfinite(values)
        elif operation == "in":
            if not isinstance(value, (list, tuple)):
                raise ValueError("The in operation requires a list of scalar values")
            comparison = np.isin(values, value)
        elif operation in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if isinstance(value, (list, dict, tuple)) or value is None:
                raise ValueError("Metadata comparisons require one non-null scalar value")
            function = {
                "eq": np.equal,
                "ne": np.not_equal,
                "lt": np.less,
                "le": np.less_equal,
                "gt": np.greater,
                "ge": np.greater_equal,
            }[operation]
            comparison = function(values, value)
        else:
            raise ValueError(f"Unsupported metadata filter operation: {operation}")
        mask &= comparison
    return mask


def compile_predicate(skeleton: Skeleton, edge: EdgeSpec) -> tuple[np.ndarray, np.ndarray]:
    """Compile only fixed, separable endpoint metadata comparisons."""
    predicate = edge.predicate
    if isinstance(predicate, str):
        if predicate not in {"path_membership", "separable_metadata"}:
            raise ValueError(f"Unsupported grounding predicate {predicate!r}")
        predicate = {}
    if not isinstance(predicate, Mapping) or set(predicate) - {"source", "destination"}:
        raise ValueError("Only separable source/destination metadata filters are supported")
    return (
        _filter_mask(skeleton, edge.source.table, predicate.get("source", [])),
        _filter_mask(skeleton, edge.target.table, predicate.get("destination", [])),
    )


def _singleton_groups(count: int, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Group number equals the source storage ordinal. Each group has one member.
    return (
        np.arange(count + 1, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        np.arange(count, dtype=np.int64),
    )


@dataclass
class GroundingPlan:
    skeleton: Skeleton
    edge: EdgeSpec
    offsets: np.ndarray
    source_rows: np.ndarray
    destination_group_ids: np.ndarray
    counters: dict[str, int]
    pattern: str

    @property
    def nbytes(self) -> int:
        return self.offsets.nbytes + self.source_rows.nbytes + self.destination_group_ids.nbytes

    @property
    def group_count(self) -> int:
        return len(self.offsets) - 1

    @property
    def empty_group(self) -> int:
        return self.group_count - 1

    def group(self, group_id: int) -> np.ndarray:
        return self.source_rows[self.offsets[group_id] : self.offsets[group_id + 1]]

    def parents(self, destination_row: int) -> np.ndarray:
        return self.group(int(self.destination_group_ids[destination_row]))

    def selected_parents(
        self,
        destination_row: int,
        policy: str = "full",
        cap: int = 8,
        seed: int = 0,
        edge_key: Any = None,
    ) -> np.ndarray:
        group = self.parents(destination_row)
        self.counters["destination_lookups"] += 1
        if policy == "full":
            selected = group
        elif policy == "bounded":
            if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
                raise ValueError("The per-edge cell parent cap must be a positive integer")
            count = min(cap, len(group))
            if count == len(group):
                selected = group
            else:
                stable_id = int(self.skeleton.row_ids[self.edge.target.table][destination_row])
                identity = (
                    edge_key
                    if edge_key is not None
                    else (
                        self.edge.source.identity,
                        self.edge.target.identity,
                        tuple((step.role.identity, step.direction) for step in self.edge.path),
                    )
                )
                rng = np.random.default_rng(keyed_seed(seed, "cell-parents", identity, stable_id))
                selected = group[sample_positions(len(group), count, rng)]
            self.counters["sampled_positions"] += count
        else:
            raise ValueError(f"Unknown parent policy {policy!r}")
        self.counters["source_id_reads"] += len(selected)
        return selected

    def pool_group_embeddings(
        self,
        encoded: torch.Tensor,
        pooling: str | None = None,
        temp_memory_bytes: int = DEFAULT_TEMP_BYTES,
    ) -> torch.Tensor:
        """Pool each reusable neighborhood once, preserving endpoint counts."""
        pooling = pooling or self.edge.pooling
        if pooling not in {"mean", "sum"}:
            raise ValueError("Row pooling must be mean or sum")
        if encoded.ndim != 2 or len(encoded) != self.skeleton.row_counts[self.edge.source.table]:
            raise ValueError("Encoded source values must have shape (source_rows, embedding_dim)")
        dimensions = encoded.shape[1]
        # Large float32 groups otherwise lose precision during sequential sums.
        # Return the encoder dtype after accumulating in float64.
        accumulator_dtype = (
            torch.float64
            if encoded.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else encoded.dtype
        )
        accumulator_size = torch.empty((), dtype=accumulator_dtype).element_size()
        result_bytes = self.group_count * (
            dimensions * (accumulator_size + encoded.element_size()) + accumulator_size
        )
        check_memory(result_bytes, temp_memory_bytes, "Exact pooled group embeddings")
        pooled = torch.zeros(
            (self.group_count, dimensions), dtype=accumulator_dtype, device=encoded.device
        )
        available = temp_memory_bytes - result_bytes
        bytes_per_visit = max(1, dimensions * (encoded.element_size() + accumulator_size) + 16)
        if len(self.source_rows):
            check_memory(
                result_bytes + bytes_per_visit, temp_memory_bytes, "Exact pooling workspace"
            )
        batch = max(1, available // bytes_per_visit)
        for start in range(0, len(self.source_rows), batch):
            stop = min(start + batch, len(self.source_rows))
            positions = np.arange(start, stop, dtype=np.int64)
            groups = np.searchsorted(self.offsets, positions, side="right") - 1
            group_tensor = torch.as_tensor(groups, dtype=torch.long, device=encoded.device)
            row_tensor = torch.as_tensor(
                self.source_rows[start:stop].copy(), dtype=torch.long, device=encoded.device
            )
            pooled.index_add_(0, group_tensor, encoded[row_tensor].to(accumulator_dtype))
            self.counters["group_pool_visits"] += stop - start
        if pooling == "mean":
            counts = torch.as_tensor(
                np.diff(self.offsets).copy(), dtype=accumulator_dtype, device=encoded.device
            )
            pooled /= counts.clamp_min(1).unsqueeze(-1)
        return pooled.to(encoded.dtype)

    def prepare_pool(
        self,
        encoded: torch.Tensor,
        policy: str = "full",
        cap: int = 8,
        seed: int = 0,
        edge_key: Any = None,
        pooling: str | None = None,
        temp_memory_bytes: int = DEFAULT_TEMP_BYTES,
        parent_map_memory_bytes: int = DEFAULT_INDEX_BYTES,
    ) -> PreparedPool:
        if policy not in {"full", "bounded"}:
            raise ValueError(f"Unknown parent policy {policy!r}")
        pooling = pooling or self.edge.pooling
        if pooling not in {"mean", "sum"}:
            raise ValueError("Row pooling must be mean or sum")
        if encoded.ndim != 2 or len(encoded) != self.skeleton.row_counts[self.edge.source.table]:
            raise ValueError("Encoded source values must have shape (source_rows, embedding_dim)")
        group_values = (
            self.pool_group_embeddings(encoded, pooling, temp_memory_bytes)
            if policy == "full"
            else None
        )
        # Exact pools no longer need the entire encoded source buffer. Retain
        # only its tensor metadata so callers can release that large allocation.
        if policy == "full":
            encoded = encoded.new_empty((0, encoded.shape[1]))
        return PreparedPool(
            self,
            encoded,
            group_values,
            policy,
            cap,
            seed,
            edge_key,
            pooling,
            temp_memory_bytes,
            parent_map_memory_bytes,
        )

    def pool(
        self,
        encoded: torch.Tensor,
        rows: Any = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.prepare_pool(encoded, **kwargs).gather(rows)

    def pool_chunk(self, encoded: torch.Tensor, start: int, end: int, **kwargs) -> torch.Tensor:
        """For repeated full-policy chunks, reuse prepare_pool().gather instead."""
        return self.pool(encoded, slice(start, end), **kwargs)

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(
            {
                "source": self.edge.source.identity,
                "target": self.edge.target.identity,
                "path": [(step.role.identity, step.direction) for step in self.edge.path],
                "predicate": self.edge.predicate,
                "groups": [
                    [
                        int(self.skeleton.row_ids[self.edge.source.table][row])
                        for row in self.group(group)
                    ]
                    for group in range(self.group_count)
                ],
                "destinations": sorted(
                    zip(
                        self.skeleton.row_ids[self.edge.target.table].tolist(),
                        self.destination_group_ids.tolist(),
                    )
                ),
            }
        )


@dataclass
class PreparedPool:
    plan: GroundingPlan
    encoded: torch.Tensor
    group_values: torch.Tensor | None
    policy: str
    cap: int
    seed: int
    edge_key: Any
    pooling: str
    temp_memory_bytes: int
    parent_map_memory_bytes: int

    def gather(self, rows: Any = None) -> torch.Tensor:
        count = len(self.plan.destination_group_ids)
        if rows is None:
            rows = np.arange(count, dtype=np.int64)
        elif isinstance(rows, slice):
            start, stop, step = rows.indices(count)
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if self.policy == "full":
            required = (
                (len(rows) + len(self.group_values))
                * self.encoded.shape[1]
                * self.encoded.element_size()
            )
            check_memory(required, self.temp_memory_bytes, "Destination pooled embeddings")
            groups = self.plan.destination_group_ids[rows]
            self.plan.counters["destination_lookups"] += len(rows)
            index = torch.as_tensor(groups.copy(), dtype=torch.long, device=self.encoded.device)
            return self.group_values[index]
        if not isinstance(self.cap, int) or isinstance(self.cap, bool) or self.cap <= 0:
            raise ValueError("The per-edge cell parent cap must be a positive integer")
        dimensions = self.encoded.shape[1]
        # A bound is checked before allocating maps or embedding gathers.
        width = min(self.cap, self.plan.skeleton.row_counts[self.plan.edge.source.table])
        check_memory(
            len(rows) * (width * 8 + 8), self.parent_map_memory_bytes, "Bounded parent map chunk"
        )
        required = len(rows) * (
            width * (dimensions * self.encoded.element_size() + 9)
            + dimensions * self.encoded.element_size()
        )
        check_memory(required, self.temp_memory_bytes, "Bounded pooling workspace")
        result = self.encoded.new_zeros((len(rows), dimensions))
        if width == 0 or not len(rows):
            return result
        parents = np.zeros((len(rows), width), dtype=np.int64)
        counts = np.zeros(len(rows), dtype=np.int64)
        for index, row in enumerate(rows):
            selected = self.plan.selected_parents(
                int(row), "bounded", self.cap, self.seed, self.edge_key
            )
            counts[index] = len(selected)
            parents[index, : len(selected)] = selected
        parent_tensor = torch.as_tensor(parents, dtype=torch.long, device=self.encoded.device)
        mask = torch.arange(width, device=self.encoded.device).unsqueeze(0) < torch.as_tensor(
            counts, device=self.encoded.device
        ).unsqueeze(-1)
        embeddings = self.encoded[parent_tensor]
        embeddings.masked_fill_(~mask.unsqueeze(-1), 0)
        result = embeddings.sum(dim=1)
        if self.pooling == "mean":
            result /= (
                torch.as_tensor(counts, dtype=self.encoded.dtype, device=self.encoded.device)
                .clamp_min(1)
                .unsqueeze(-1)
            )
        return result


def compile_grounding(
    skeleton: Skeleton,
    edge: EdgeSpec,
    index_memory_bytes: int | None = None,
    source_filter: np.ndarray | None = None,
    destination_filter: np.ndarray | None = None,
) -> GroundingPlan:
    """Resolve all supported traversal patterns without candidate-pair expansion."""
    source, target = edge.source.table, edge.target.table
    if source not in skeleton.row_counts or target not in skeleton.row_counts:
        raise ValueError("Grounding edge references an unknown table")
    path = tuple(edge.path)
    if len(path) > 2:
        raise ValueError("Only one- and two-hop paths are supported")
    current = target
    visited = {current}
    for step in path:
        if step.role not in skeleton.fk_assignments:
            raise ValueError(f"Grounding path uses unknown FK role {step.role.key}")
        if step.start_table != current:
            raise ValueError("Grounding path is not contiguous from destination to source")
        current = step.end_table
        if current in visited:
            raise ValueError("Only simple table paths are supported")
        visited.add(current)
    if current != source or (source != target and not path) or (source == target and path):
        raise ValueError(
            "Within-table edges require same-row identity; cross-table paths must reach their source table"
        )
    if edge.source == edge.target:
        raise ValueError("Same-column cross-row dependencies are unsupported")
    source_count, target_count = skeleton.row_counts[source], skeleton.row_counts[target]
    source_mask, target_mask = compile_predicate(skeleton, edge)
    for explicit, mask, count, label in (
        (source_filter, source_mask, source_count, "source"),
        (destination_filter, target_mask, target_count, "destination"),
    ):
        if explicit is not None:
            explicit = np.asarray(explicit)
            if explicit.shape != (count,) or explicit.dtype.kind != "b":
                raise ValueError(f"The {label} filter must be a Boolean metadata mask")
            mask &= explicit
    limit = index_memory_bytes if index_memory_bytes is not None else skeleton.index_memory_bytes
    # Each pattern stores at most one source endpoint per observed intermediate
    # witness, plus source/destination singleton maps. Reserve before construction.
    witness_bound = source_count
    if (
        path
        and path[0].direction == "pk_to_fk"
        and len(path) == 2
        and path[1].direction == "fk_to_pk"
    ):
        witness_bound = skeleton.row_counts[path[0].end_table]
    group_bound = max(
        [source_count, target_count] + [skeleton.row_counts[step.start_table] for step in path]
    )
    check_memory(
        (witness_bound + target_count + group_bound + 3) * 8, limit, "Grounded endpoint indexes"
    )
    index_visits = 0
    if not path:
        offsets, members, _ = _singleton_groups(source_count, skeleton.row_ids[source])
        destinations = np.arange(target_count, dtype=np.int64)
        pattern = "same_row"
    elif len(path) == 1 and path[0].direction == "fk_to_pk":
        offsets, members, _ = _singleton_groups(source_count, skeleton.row_ids[source])
        destinations = skeleton.forward(path[0].role).copy()
        pattern = "forward"
    elif len(path) == 1:
        inverse = skeleton.inverse(path[0].role)
        offsets, members = inverse.offsets, inverse.child_rows
        destinations = np.arange(target_count, dtype=np.int64)
        pattern = "reverse"
    elif path[0].direction == "fk_to_pk" and path[1].direction == "fk_to_pk":
        offsets, members, _ = _singleton_groups(source_count, skeleton.row_ids[source])
        first, second = skeleton.forward(path[0].role), skeleton.forward(path[1].role)
        destinations = np.full(target_count, -1, dtype=np.int64)
        valid = first >= 0
        destinations[valid] = second[first[valid]]
        index_visits = target_count
        pattern = "forward_forward"
    elif path[0].direction == "fk_to_pk" and path[1].direction == "pk_to_fk":
        inverse = skeleton.inverse(path[1].role)
        offsets, members = inverse.offsets, inverse.child_rows
        destinations = skeleton.forward(path[0].role).copy()
        pattern = "forward_reverse_shared_pivot"
    elif path[0].direction == "pk_to_fk" and path[1].direction == "fk_to_pk":
        first, second = skeleton.inverse(path[0].role), skeleton.forward(path[1].role)
        offsets = np.empty(target_count + 1, dtype=np.int64)
        offsets[0] = 0
        pieces = []
        for row in range(target_count):
            pivots = first.group(row)
            candidates = second[pivots]
            candidates = np.unique(candidates[candidates >= 0])
            candidates = candidates[np.argsort(skeleton.row_ids[source][candidates], kind="stable")]
            pieces.append(candidates)
            offsets[row + 1] = offsets[row] + len(candidates)
            index_visits += len(pivots)
        members = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        destinations = np.arange(target_count, dtype=np.int64)
        pattern = "reverse_forward_unique"
    else:
        # Each source has exactly one pivot and each pivot exactly one
        # destination, so compose the scalar maps and invert once.
        first, second = skeleton.forward(path[0].role), skeleton.forward(path[1].role)
        source_to_target = np.full(source_count, -1, dtype=np.int64)
        valid = second >= 0
        source_to_target[valid] = first[second[valid]]
        inverse = inverse_index(source_to_target, target_count, skeleton.row_ids[source])
        offsets, members = inverse.offsets, inverse.child_rows
        destinations = np.arange(target_count, dtype=np.int64)
        index_visits = source_count
        pattern = "reverse_reverse"
    # Source eligibility is applied once to reusable groups, never per target.
    if not np.all(source_mask):
        retained = source_mask[members]
        old_positions = np.flatnonzero(retained)
        group_ids = np.searchsorted(offsets, old_positions, side="right") - 1
        counts = np.bincount(group_ids, minlength=len(offsets) - 1)
        members = members[retained]
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        index_visits += len(retained)
    # Append a real empty group for all null/masked destinations.
    group_count = len(offsets) - 1
    offsets = np.concatenate((offsets, [offsets[-1]])).astype(np.int64)
    members = np.asarray(members, dtype=np.int64)
    destinations[(destinations < 0) | ~target_mask] = group_count
    for array in (offsets, members, destinations):
        array.setflags(write=False)
    plan = GroundingPlan(
        skeleton,
        edge,
        offsets,
        members,
        destinations,
        {
            "source_index_visits": index_visits,
            "source_id_reads": 0,
            "destination_lookups": 0,
            "group_pool_visits": 0,
            "sampled_positions": 0,
        },
        pattern,
    )
    check_memory(plan.nbytes, limit, "Grounded endpoint indexes")
    return plan
