"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        # Exact 2-step lookahead: pick the candidate that minimizes
        # d(current, a) + d(a, b) + d(b, dest) vs d(current, b) + d(b, a) + d(a, dest)
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return int(a if cost_a <= cost_b else b)

    if n_unvisited == 3:
        # Exact 3-step lookahead via brute force over 3! = 6 permutations
        nodes = unvisited_nodes
        best_cost = np.inf
        best_first = nodes[0]
        import itertools
        for perm in itertools.permutations(range(3)):
            p = [nodes[i] for i in perm]
            cost = (distance_matrix[current_node, p[0]] +
                    distance_matrix[p[0], p[1]] +
                    distance_matrix[p[1], p[2]] +
                    distance_matrix[p[2], destination_node])
            if cost < best_cost:
                best_cost = cost
                best_first = p[0]
        return int(best_first)

    # For larger sets, use composite heuristic with local density metric
    # Check time budget; if tight, fall back to simple nearest-neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Vectorized composite scoring
    # 1. Distance from current to each unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]  # shape (n,)

    # 2. Distance from each unvisited to destination (normalized)
    d_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n,)
    # Normalize to [0, 1] range for weighting
    d_dest_range = d_dest.max() - d_dest.min()
    if d_dest_range > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_range
    else:
        d_dest_norm = np.zeros(n_unvisited)

    # 3. Nearest-neighbor gap: for each unvisited node, min distance to any OTHER unvisited node
    # This measures local integration/isolation
    # Build submatrix of distances among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # shape (n, n)
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(sub_dist, np.inf)
    nn_gap = sub_dist.min(axis=1)  # shape (n,) - nearest neighbor distance for each node

    # Normalize gap to [0, 1]
    gap_range = nn_gap.max() - nn_gap.min()
    if gap_range > 1e-12:
        gap_norm = (nn_gap - nn_gap.min()) / gap_range
    else:
        gap_norm = np.zeros(n_unvisited)

    # Composite score: prefer low d_current, low d_dest_norm (close to dest is fine, 
    # but we want to avoid leaving dest too far), and low gap_norm (well-integrated)
    # Weights: primary is d_current, secondary is gap (local density), tertiary is dest
    alpha = 0.15  # weight for destination proximity
    beta = 0.35   # weight for local integration (gap)

    # Normalize d_current to [0,1] for consistent scaling
    dc_range = d_current.max() - d_current.min()
    if dc_range > 1e-12:
        d_current_norm = (d_current - d_current.min()) / dc_range
    else:
        d_current_norm = np.zeros(n_unvisited)

    score = d_current_norm + alpha * d_dest_norm + beta * gap_norm
    idx = np.argmin(score)
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
