"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # 5s safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _simulate_total(capacity, weights, values, order, skip_idx):
    """Simulate greedy fill following `order`, skipping `skip_idx`. Return total value."""
    cap = capacity
    val = 0.0
    for i in order:
        if i == skip_idx:
            continue
        w = weights[i]
        if w <= cap:
            cap -= w
            val += values[i]
        if cap <= 0:
            break
    return val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # If time is running out, fall back to simple density greedy
    if not _elapsed_ok():
        return int(np.argmax(remaining_values / remaining_weights))

    # Compute density-based order (the baseline greedy order)
    densities = remaining_values / remaining_weights
    order = np.argsort(-densities)  # descending density

    # 1-step lookahead: for each candidate item i, simulate the rest of the
    # greedy fill (following density order) starting *after* picking item i.
    # The total value if we pick i now = values[i] + simulated_remaining.
    # Pick the i that maximizes this total.
    best_score = -1.0
    best_idx = order[0]  # default: top density item

    for i in order:
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue  # infeasible
        # Simulate: pick i, then continue greedy on remaining capacity
        sim_val = remaining_values[i] + _simulate_total(
            remaining_capacity - w_i,
            remaining_weights,
            remaining_values,
            order,
            i
        )
        if sim_val > best_score:
            best_score = sim_val
            best_idx = i

        # Time check during lookahead
        if not _elapsed_ok():
            break

    return int(best_idx)
# EVOLVE-BLOCK-END
