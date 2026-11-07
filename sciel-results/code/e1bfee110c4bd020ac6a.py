"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n <= 1:
        return edge_distance.copy()

    # Normalize usage: edges in the current tour have the highest counts.
    max_used = float(edge_n_used.max())
    if max_used <= 0:
        # No usage yet; return a mild perturbation to kick things off.
        # Slightly inflate short edges to encourage exploration.
        d = edge_distance.copy()
        np.fill_diagonal(d, 0.0)
        # Gentle inverse-distance perturbation on off-diagonal
        off = d > 0
        d[off] *= (1.0 + 0.05 / np.maximum(d[off], 1e-6))
        return d

    # Normalize usage to [0, 1]
    usage_norm = edge_n_used / max_used

    # Adaptive penalty strength: grows with number of distinct used edges
    # (proxy for how long we've been running / how stuck we are).
    n_used = int((edge_n_used > 0).sum())
    # Start moderate, grow to ~1.5 for heavy usage
    alpha = 0.3 + 0.5 * min(1.0, n_used / (2.0 * n))

    # Inverse-distance weighting: penalize short edges more strongly
    # so 2-opt is pushed to break them and explore alternatives.
    d = edge_distance.copy()
    np.fill_diagonal(d, 0.0)
    off = d > 0
    # inv_dist term: 1/d for short edges, capped to avoid blowup
    inv = 1.0 / np.maximum(d[off], 0.05)
    inv = np.clip(inv, 0.0, 5.0)
    # Normalize inv to [0,1]
    inv_max = inv.max()
    if inv_max > 0:
        inv = inv / inv_max

    # Combined penalty multiplier
    # Base: usage-driven, amplified on short edges
    penalty = 1.0 + alpha * usage_norm[off] * (1.0 + 0.5 * inv)

    d[off] = d[off] * penalty

    # Ensure finiteness
    d = np.where(np.isfinite(d), d, edge_distance)
    np.fill_diagonal(d, 0.0)
    return d
# EVOLVE-BLOCK-END
