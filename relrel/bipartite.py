"""Sample v1.0.0 HSBM row references without constructing a NetworkX cell graph."""

import numpy as np

from relrel.utils import assign_cluster_at_levels, get_probs_at_levels


def sample_bipartite_assignments(
    size_a: int,
    size_b: int,
    hierarchy_a: list[int],
    hierarchy_b: list[int],
    chunk_memory_bytes: int = 256 * 1024**2,
) -> np.ndarray:
    """Return one parent ordinal per child, using the original block affinities.

    Child rows in the same leaf block share a categorical probability vector.
    Reusing that vector avoids quadratic Python loops and parent/child graph
    objects. Uniform draws remain in child-row order. No FK-null or additional
    attractiveness prior is introduced; imported skeletons can contain nulls.
    """
    if size_a < 0 or size_b < 0:
        raise ValueError("Row counts must be nonnegative")
    if len(hierarchy_a) != len(hierarchy_b) or not hierarchy_a:
        raise ValueError("HSBM hierarchies must have the same positive depth")
    if any(type(n) not in (int, np.int64, np.int32) or n < 1 for n in (*hierarchy_a, *hierarchy_b)):
        raise ValueError("HSBM branching factors must be positive integers")
    if not size_a or not size_b:
        return np.full(size_b, -1, dtype=np.int64)
    if chunk_memory_bytes < size_a * 8 * 3:
        raise MemoryError("FK assignment workspace cannot hold one parent probability vector")
    a = assign_cluster_at_levels(size_a, hierarchy_a)
    b = assign_cluster_at_levels(size_b, hierarchy_b)
    probabilities = get_probs_at_levels(hierarchy_a, hierarchy_b)
    draws = np.random.random(size_b)
    assignments = np.empty(size_b, dtype=np.int64)
    # Membership is contiguous in v1.0.0, including when late blocks are empty.
    starts = np.r_[0, np.flatnonzero(np.any(b[1:] != b[:-1], axis=1)) + 1, size_b]
    for start, stop in zip(starts[:-1], starts[1:]):
        weights = np.ones(size_a, dtype=np.float64)
        for level, matrix in enumerate(probabilities):
            weights *= matrix[a[:, level], b[start, level]]
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("HSBM affinities must have positive finite total mass")
        cdf = np.cumsum(weights / total)
        cdf[-1] = 1.0
        assignments[start:stop] = np.searchsorted(cdf, draws[start:stop], side="right")
    return assignments
