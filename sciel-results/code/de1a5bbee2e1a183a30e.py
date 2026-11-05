"""Priority scorer for Maximum Independent Set (greedy decoder)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _fallback_priority(adjacency):
    # Seed behavior: lowest degree first.
    return -adjacency.sum(axis=1).astype(float)


def priority(adjacency):
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock limit

    try:
        A = np.asarray(adjacency, dtype=np.float64)
    except Exception:
        return _fallback_priority(adjacency)

    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    # Symmetrize and clear any diagonal (self-loops are irrelevant to MIS).
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)

    # Cost of taking a vertex = its own degree (blocks its neighbors) plus the
    # degrees of its neighbors (each neighbor is a candidate we forgo).
    # Small cost => "cheap" to include => higher priority.
    cost = deg + A @ deg

    # Primary: lower cost first.  Tie-break: lower degree first.
    score = -cost + 1e-6 * deg

    if time.time() > deadline:
        return _fallback_priority(adjacency)

    return score
# EVOLVE-BLOCK-END
