"""Global feature-column DAG contracts and schema-local proposal sampling.

The column DAG determines causal order. Foreign keys provide row paths and
locality constraints, not causal direction or a table-completion schedule.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np

from relrel.config import DAGParams, RSCMParams, SCMParams
from relrel.dag import DAG_REGISTRY
from relrel.rng import config_fingerprint, keyed_seed, scoped_rng

GRAPH_VERSION = "global-column-dag-v1"


@dataclass(frozen=True, order=True)
class Column:
    table: str
    name: str

    def __post_init__(self):
        if (
            not isinstance(self.table, str)
            or not self.table
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise ValueError("A column requires non-empty table and feature names")

    @property
    def identity(self) -> tuple[str, str]:
        return self.table, self.name

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.name}"

    def to_dict(self) -> dict[str, str]:
        return {"table": self.table, "name": self.name}


@dataclass(frozen=True, order=True)
class FKRole:
    """A named scalar-or-null reference from a child to a distinct parent."""

    name: str
    child_table: str
    parent_table: str

    def __post_init__(self):
        if not all(
            isinstance(x, str) and x for x in (self.name, self.child_table, self.parent_table)
        ):
            raise ValueError("FK names and tables must be non-empty strings")
        if self.child_table == self.parent_table:
            raise ValueError("Self-referential foreign keys are unsupported")

    @property
    def identity(self) -> tuple[str, str]:
        return self.child_table, self.name

    @property
    def key(self) -> str:
        return f"{self.child_table}.{self.name}"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "child_table": self.child_table,
            "parent_table": self.parent_table,
        }


@dataclass(frozen=True, order=True)
class PathStep:
    role: FKRole
    direction: Literal["fk_to_pk", "pk_to_fk"]

    def __post_init__(self):
        if self.direction not in ("fk_to_pk", "pk_to_fk"):
            raise ValueError(f"Unsupported FK traversal direction: {self.direction}")

    @property
    def start_table(self) -> str:
        return self.role.child_table if self.direction == "fk_to_pk" else self.role.parent_table

    @property
    def end_table(self) -> str:
        return self.role.parent_table if self.direction == "fk_to_pk" else self.role.child_table

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.to_dict(), "direction": self.direction}


@dataclass(frozen=True)
class EdgeSpec:
    source: Column
    target: Column
    path: tuple[PathStep, ...] = ()
    weight: float = 1.0
    pooling: str = "mean"
    predicate: str | Mapping[str, Any] = "path_membership"
    encoder_key: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "path", tuple(self.path))
        if self.source == self.target:
            raise ValueError("Same-column cross-row dependencies and self edges are unsupported")
        if not math.isfinite(float(self.weight)):
            raise ValueError("Edge weights must be finite")
        if self.pooling not in ("mean", "sum"):
            raise ValueError("Pooling must be mean or sum")
        if isinstance(self.predicate, str):
            if self.predicate not in ("path_membership", "separable_metadata"):
                raise ValueError("Unsupported predicate mode")
        elif not isinstance(self.predicate, Mapping):
            raise ValueError("Predicates must be structural names or metadata filter mappings")
        if self.source.table == self.target.table:
            if self.path:
                raise ValueError("Same-table column dependencies use same-row identity")
        else:
            if len(self.path) not in (1, 2):
                raise ValueError("Cross-table paths must contain one or two FK steps")
            tables = [self.target.table]
            for step in self.path:
                if step.start_table != tables[-1]:
                    raise ValueError("Path steps must connect from target toward source")
                tables.append(step.end_table)
            if tables[-1] != self.source.table or len(set(tables)) != len(tables):
                raise ValueError("A path must be simple and end in the source table")

    @property
    def identity(self) -> tuple[tuple[str, str], tuple[str, str]]:
        return self.source.identity, self.target.identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "path": [step.to_dict() for step in self.path],
            "weight": float(self.weight),
            "pooling": self.pooling,
            "predicate": self.predicate,
            "encoder_key": self.encoder_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EdgeSpec":
        return cls(
            source=Column(**value["source"]),
            target=Column(**value["target"]),
            path=tuple(
                PathStep(FKRole(**step["role"]), step["direction"])
                for step in value.get("path", ())
            ),
            weight=value.get("weight", 1.0),
            pooling=value.get("pooling", "mean"),
            predicate=value.get("predicate", "path_membership"),
            encoder_key=value.get("encoder_key"),
        )


@dataclass(frozen=True)
class ColumnGraph:
    columns: tuple[Column, ...]
    edges: tuple[EdgeSpec, ...]
    order: tuple[Column, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "order", tuple(self.order))
        self.validate()

    def incoming(self, column: Column) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.target == column)

    def outgoing(self, column: Column) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.source == column)

    def validate(
        self,
        fk_roles: Sequence[FKRole] | None = None,
        max_hops: int | None = None,
        column_parent_cap: int | None = None,
    ):
        vertices = set(self.columns)
        if len(vertices) != len(self.columns):
            raise ValueError("Duplicate qualified feature columns")
        if len(self.order) != len(vertices) or set(self.order) != vertices:
            raise ValueError("Topological order must contain every column exactly once")
        rank = {column: i for i, column in enumerate(self.order)}
        seen = set()
        for edge in self.edges:
            if edge.source not in vertices or edge.target not in vertices:
                raise ValueError("Every edge endpoint must be a feature column")
            if edge.identity in seen:
                raise ValueError("Duplicate column edge")
            seen.add(edge.identity)
            if rank[edge.source] >= rank[edge.target]:
                raise ValueError("Every edge must increase column topological rank")
            if max_hops is not None and len(edge.path) > max_hops:
                raise ValueError("Edge exceeds schema locality limit")
            if fk_roles is not None and any(step.role not in fk_roles for step in edge.path):
                raise ValueError("Path references an unknown FK role")
        if column_parent_cap is not None and any(
            len(self.incoming(v)) > column_parent_cap for v in vertices
        ):
            raise ValueError("Column-parent budget exceeded")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": GRAPH_VERSION,
            "columns": [v.to_dict() for v in self.columns],
            "edges": [edge.to_dict() for edge in self.edges],
            "order": [v.to_dict() for v in self.order],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ColumnGraph":
        if value.get("version", GRAPH_VERSION) != GRAPH_VERSION:
            raise ValueError("Unsupported column graph version")
        return cls(
            tuple(Column(**v) for v in value["columns"]),
            tuple(EdgeSpec.from_dict(edge) for edge in value["edges"]),
            tuple(Column(**v) for v in value["order"]),
            value.get("metadata", {}),
        )

    def fingerprint(self) -> str:
        return config_fingerprint(self.to_dict())


def enumerate_schema_paths(
    start_table: str, end_table: str, fk_roles: Sequence[FKRole], max_hops: int = 2
) -> tuple[tuple[PathStep, ...], ...]:
    """Enumerate named, simple paths from destination to source table."""
    if max_hops not in (1, 2):
        raise ValueError("Only one- and two-hop paths are supported")
    if start_table == end_table:
        return ((),)
    roles = sorted(fk_roles)
    if len({role.identity for role in roles}) != len(roles):
        raise ValueError("FK role identities must be unique")
    adjacency: dict[str, list[PathStep]] = {}
    for role in roles:
        for direction in ("fk_to_pk", "pk_to_fk"):
            step = PathStep(role, direction)
            adjacency.setdefault(step.start_table, []).append(step)
    paths = []

    def visit(table: str, path: tuple[PathStep, ...], visited: frozenset[str]):
        if len(path) == max_hops:
            return
        for step in adjacency.get(table, ()):
            if step.end_table in visited:
                continue
            extended = (*path, step)
            if step.end_table == end_table:
                paths.append(extended)
            else:
                visit(step.end_table, extended, visited | {step.end_table})

    visit(start_table, (), frozenset({start_table}))
    return tuple(paths)


def sample_column_graph(
    columns: Sequence[Column] | Mapping[str, Sequence[str]],
    fk_roles: Sequence[FKRole],
    scm_params: SCMParams,
    dag_params: DAGParams,
    rscm_params: RSCMParams,
    seed: int = 0,
    *,
    proposal: nx.DiGraph | None = None,
    column_assignment: Sequence[Column] | None = None,
) -> ColumnGraph:
    """Sample a proposal, associate feature identities, mask, cap and order.

    Incoming edges use independent keyed priorities before capping. Thus a
    locality change never resamples the proposal, assignment or priorities.
    ``proposal`` and ``column_assignment`` support frozen structural fixtures.
    No connectivity repair is performed after the locality mask.
    """
    if isinstance(columns, Mapping):
        columns = [Column(table, name) for table, names in columns.items() for name in names]
    columns = tuple(sorted(columns))
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate qualified feature columns")
    if len({role.identity for role in fk_roles}) != len(fk_roles):
        raise ValueError("FK role identities must be unique")
    n = len(columns)
    proposal_seed = keyed_seed(seed, "column-proposal")
    family = "frozen"
    if proposal is None:
        if n < 2:
            proposal = nx.DiGraph()
            proposal.add_nodes_from(range(n))
            family = "isolated"
        elif n == 2:
            # BA(m=2) cannot construct a two-vertex proposal. SQL schemas can
            # legitimately have only two feature columns; no hidden buffer is
            # added to a graph whose vertices are the observed inventory.
            with scoped_rng(proposal_seed):
                proposal = nx.DiGraph([(0, 1, {"weight": float(np.random.randn())})])
            family = "two_column"
        else:
            with scoped_rng(proposal_seed):
                family = str(scm_params.scm_layout_choices.sample_uniform())
                proposal = DAG_REGISTRY[family](
                    num_nodes=n, dag_params=dag_params, seed=proposal_seed % (1 << 32)
                ).graph.copy()
    else:
        proposal = proposal.copy()
    # Family implementations are allowed to construct their graph from edges.
    # Retain the complete inventory even if that dropped isolated vertices.
    if any(node not in range(n) for node in proposal.nodes):
        raise ValueError("Proposal node IDs must be integers in the feature inventory")
    proposal.add_nodes_from(range(n))
    if not nx.is_directed_acyclic_graph(proposal):
        raise ValueError("The global column proposal must be a DAG")
    if column_assignment is None:
        permutation = np.random.default_rng(keyed_seed(seed, "column-association")).permutation(n)
        assignment = tuple(columns[int(i)] for i in permutation)
    else:
        assignment = tuple(column_assignment)
        if len(assignment) != n or set(assignment) != set(columns):
            raise ValueError("Column association must be a permutation of the feature inventory")

    path_cache = {}
    candidates: dict[Column, list[EdgeSpec]] = {}
    proposed_edges = []
    masked = 0
    for u, v, data in sorted(proposal.edges(data=True)):
        source, target = assignment[u], assignment[v]
        weight = float(data.get("weight", 1.0))
        if not math.isfinite(weight):
            raise ValueError("Proposal weights must be finite")
        proposed_edges.append([int(u), int(v), weight])
        key = (target.table, source.table)
        if key not in path_cache:
            path_cache[key] = enumerate_schema_paths(*key, fk_roles, rscm_params.max_hops)
        paths = path_cache[key]
        if not paths:
            masked += 1
            continue
        rng = np.random.default_rng(keyed_seed(seed, "edge-path", source.identity, target.identity))
        path = paths[int(rng.integers(len(paths)))]
        edge = EdgeSpec(
            source,
            target,
            path,
            weight,
            rscm_params.row_pooling,
            encoder_key=config_fingerprint(["encoder", source.identity, target.identity]),
        )
        candidates.setdefault(target, []).append(edge)
    edges = []
    cap_removed = 0
    for target, incoming in sorted(candidates.items()):
        incoming.sort(
            key=lambda edge: (
                keyed_seed(seed, "column-parent-priority", edge.identity),
                edge.source,
            )
        )
        cap_removed += max(0, len(incoming) - rscm_params.column_parent_cap)
        edges.extend(incoming[: rscm_params.column_parent_cap])
    edges.sort(key=lambda edge: (edge.source, edge.target))
    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    graph.add_edges_from((edge.source, edge.target) for edge in edges)
    order = tuple(nx.lexicographical_topological_sort(graph, key=lambda column: column))
    result = ColumnGraph(
        columns,
        tuple(edges),
        order,
        {
            "sampler_version": GRAPH_VERSION,
            "seed": int(seed),
            "proposal_seed": proposal_seed,
            "proposal_family": family,
            "proposal_edges": proposed_edges,
            "column_assignment": [column.to_dict() for column in assignment],
            "locality_masked_edges": masked,
            "parent_cap_removed_edges": cap_removed,
            "parent_cap_policy": "independent_keyed_priority",
            "max_hops": rscm_params.max_hops,
            "column_parent_cap": rscm_params.column_parent_cap,
        },
    )
    result.validate(fk_roles, rscm_params.max_hops, rscm_params.column_parent_cap)
    return result
