"""Structural change: 2-step lookahead with geometric detour penalty and adaptive destination pull."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin within 150s wall-clock kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using a 2-step lookahead heuristic with geometric detour penalty.
    
    For each candidate j, we estimate the cost of:
      - moving current -> j
      - then moving j -> the best next node k (or destination if j is last)
    
    Additionally, we apply a geometric detour penalty: if j causes a large
    detour away from the straight-line path from current to destination,
    we penalize it. This helps avoid zigzagging.
    
    As the tour nears completion, we blend in pure destination-proximity
    to ensure the final segment is short.
    
    Parameters
    ----------
    current_node : int
    destination_node : int
    unvisited_nodes : np.ndarray
    distance_matrix : np.ndarray
    Returns
    -------
    int : index of the next node to visit
    """
    unvisited = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited)
    d = distance_matrix
    n_total = d.shape[0]

    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited[0])

    # Deadline guard: fall back to pure nearest-neighbor
    if time.time() > _DEADLINE - 2.0:
        d_curr = d[current_node, unvisited]
        return int(unvisited[int(np.argmin(d_curr))])

    # Precompute distances
    d_curr = d[current_node, unvisited]  # shape (m,)
    d_dest = d[unvisited, destination_node]  # shape (m,)
    d_direct = d[current_node, destination_node]

    # 2-step lookahead: for each candidate j, find the best k (or destination)
    # to visit after j
    m = n_unvisited
    # d[j, k] for all j, k in unvisited
    d_unvisited = d[np.ix_(unvisited, unvisited)]  # shape (m, m)

    # If j is the last node, the next step is to destination
    # Otherwise, the next step is to the closest unvisited node k != j
    # Cost of choosing j: d(curr, j) + min_{k != j} d(j, k)  (or d(j, dest) if last)

    # For efficiency, compute for each j the minimum distance to any other unvisited node
    # Set diagonal to inf to exclude self
    d_unvisited_safe = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_safe, np.inf)
    min_next_from_j = np.min(d_unvisited_safe, axis=1)  # shape (m,)

    # If only 2 unvisited nodes left, the "next" after j is destination
    if m == 2:
        lookahead_cost = d_curr + d_dest
    else:
        # Blend: for most cases, use min_next_from_j; for the last one, use d_dest
        # Actually, we don't know which will be last, so use a blend
        # Use min of (min_next_from_j, d_dest) as a conservative estimate
        lookahead_cost = d_curr + np.minimum(min_next_from_j, d_dest)

    # Geometric detour penalty
    # Compute the "straight-line" progress: how much does j move us toward destination?
    # In Euclidean space, we can approximate this using the distance matrix.
    # If d(curr, dest) > d(curr, j) + d(j, dest), then j is "off the path"
    # Detour = max(0, d(curr, j) + d(j, dest) - d(curr, dest))
    detour = np.maximum(0, d_curr + d_dest - d_direct)  # shape (m,)

    # Normalize components
    lc_min, lc_max = np.min(lookahead_cost), np.max(lookahead_cost)
    if lc_max > lc_min:
        lc_norm = (lookahead_cost - lc_min) / (lc_max - lc_min)
    else:
        lc_norm = np.zeros_like(lookahead_cost)

    detour_min, detour_max = np.min(detour), np.max(detour)
    if detour_max > detour_min:
        detour_norm = (detour - detour_min) / (detour_max - detour_min)
    else:
        detour_norm = np.zeros_like(detour)

    d_dest_max = np.max(d_dest) + 1e-9
    dest_score = d_dest / d_dest_max

    # Adaptive weights based on how many nodes remain
    frac = n_unvisited / max(n_total - 1, 1)

    # Early in the tour: weight lookahead and detour heavily
    # Late in the tour: weight destination proximity heavily
    w_lookahead = 0.5 * frac ** 0.7
    w_detour = 0.3 * frac ** 0.7
    w_dest = 1.0 - w_lookahead - w_detour
    if w_dest < 0:
        w_dest = 0.0
        total = w_lookahead + w_detour
        if total > 0:
            w_lookahead /= total
            w_detour /= total
        else:
            w_lookahead, w_detour, w_dest = 0.0, 0.0, 1.0

    # Composite score: minimize (lower is better)
    scores = w_lookahead * lc_norm + w_detour * detour_norm + w_dest * dest_score

    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
