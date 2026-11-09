"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 2.0


def _check_time():
    return (time.time() - _START_TIME) < (_DEADLINE - _MARGIN)


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    if remaining_capacity <= 0:
        return -1

    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    # Work with feasible items only
    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    nf = len(fw)

    if nf == 0:
        return -1

    # Compute value density for feasible items
    densities = fv / fw

    # Sort by decreasing density
    order = np.argsort(-densities)

    # For each candidate item, simulate what the total value would be
    # if we picked that item first, then greedily filled the rest
    best_score = -1.0
    best_idx = -1

    for i in range(nf):
        if not _check_time():
            break

        # Score: value of item i + greedy fill of remaining capacity
        # excluding item i
        sim_cap = remaining_capacity - fw[i]
        sim_val = fv[i]

        for idx in order:
            if idx == i:
                continue
            w = fw[idx]
            if w <= sim_cap:
                sim_cap -= w
                sim_val += fv[idx]
            if sim_cap <= 0:
                break

        if sim_val > best_score:
            best_score = sim_val
            best_idx = feasible_indices[i]

    return int(best_idx)
# EVOLVE-BLOCK-END
