# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    # Deadline guard: evaluator kills after 630s; keep 30s safety margin
    start_time = time.time()
    DEADLINE = start_time + 600.0

    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute clockwise edge masks for each demand
    # clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n_edges, dtype=np.int8)
        if s <= t:
            for e in range(s, t):
                mask[e] = 1
        else:
            for e in range(s, n_nodes):
                mask[e] = 1
            for e in range(0, t):
                mask[e] = 1
        cw_masks.append(mask)

    # For counter-clockwise, the mask is the complement
    ccw_masks = [1 - m for m in cw_masks]

    weights = np.array([w for _, _, w in demands], dtype=np.int64)

    # Precompute base load if all clockwise, and delta for switching to ccw
    # load_cw[e] = sum of weights of demands routed clockwise that use edge e
    # If we switch demand i to ccw, load changes by: -w_i * cw_mask_i + w_i * ccw_mask_i
    # = w_i * (ccw_mask_i - cw_mask_i)
    # Note: ccw_mask_i = 1 - cw_mask_i, so delta = w_i * (1 - 2*cw_mask_i)

    # Let's think of it differently. Let x_i = 1 if ccw, 0 if cw.
    # load[e] = sum_i w_i * (cw_mask_i[e] * (1-x_i) + ccw_mask_i[e] * x_i)
    # = sum_i w_i * cw_mask_i[e] + sum_i w_i * (ccw_mask_i[e] - cw_mask_i[e]) * x_i
    # = base_load[e] + sum_i delta_i[e] * x_i

    base_load = np.zeros(n_edges, dtype=np.int64)
    deltas = np.zeros((D, n_edges), dtype=np.int64)
    for i in range(D):
        base_load += weights[i] * cw_masks[i]
        deltas[i] = weights[i] * (ccw_masks[i] - cw_masks[i])

    # Exhaustive search over 2^15 = 32768 combinations
    best_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    for mask_int in range(1 << D):
        # Check deadline
        if time.time() > DEADLINE:
            break

        # Build routing vector
        x = np.array([(mask_int >> i) & 1 for i in range(D)], dtype=np.int64)
        # Compute load
        load = base_load + x @ deltas
        max_load = np.max(load)

        if max_load < best_load:
            best_load = max_load
            best_routing = x.astype(np.float64)

    # If we broke early, best_routing is still valid (it's the best so far)
    # Return as 0/1 array
    return best_routing

# EVOLVE-BLOCK-END
