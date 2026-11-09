"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 2.0


def _check_time():
    return (time.time() - _START_TIME) < (_DEADLINE - _MARGIN)


def _simulate_greedy(capacity, weights, values, order_indices, exclude_idx):
    """
    Simulates a greedy knapsack solution given a specific ordering of items.
    Returns the total value.
    """
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    # If no items fit, stop
    if remaining_capacity <= 0:
        return -1

    # Compute value density
    densities = remaining_values / remaining_weights

    # Order indices by decreasing density
    order = np.argsort(-densities)

    # For each candidate item, simulate what the total value would be
    # if we picked that item first, then greedily filled the rest
    best_score = -1.0
    best_idx = -1

    # We'll evaluate each candidate: pick item i, then greedily fill remaining
    for i in range(n):
        if not _check_time():
            break

        if remaining_weights[i] > remaining_capacity:
            continue

        # Score: value of item i + greedy fill of remaining capacity
        # excluding item i
        sim_cap = remaining_capacity - remaining_weights[i]
        sim_val = remaining_values[i]

        for idx in order:
            if idx == i:
                continue
            w = remaining_weights[idx]
            if w <= sim_cap:
                sim_cap -= w
                sim_val += remaining_values[idx]
            if sim_cap <= 0:
                break

        if sim_val > best_score:
            best_score = sim_val
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
