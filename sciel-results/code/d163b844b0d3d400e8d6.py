"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _greedy_complete(capacity, weights, values):
    """Complete greedily by value density; return total value."""
    if len(values) == 0 or capacity <= 0:
        return 0.0
    total = 0.0
    cap = capacity
    w = weights.copy()
    v = values.copy()
    n = len(w)
    available = np.ones(n, dtype=bool)
    count = n
    while count > 0 and cap > 0:
        if _get_elapsed() > _DEADLINE - 5.0:
            break
        feasible = available & (w <= cap)
        if not np.any(feasible):
            break
        idxs = np.where(feasible)[0]
        densities = v[idxs] / w[idxs]
        best_local = np.argmax(densities)
        best = idxs[best_local]
        total += v[best]
        cap -= w[best]
        available[best] = False
        count -= 1
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    if _get_elapsed() > _DEADLINE - 2.0:
        feasible_mask = remaining_weights <= remaining_capacity
        if not np.any(feasible_mask):
            return -1
        feasible_idx = np.where(feasible_mask)[0]
        fw = remaining_weights[feasible_idx]
        fv = remaining_values[feasible_idx]
        densities = fv / fw
        return int(feasible_idx[np.argmax(densities)])

    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_idx = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_idx]
    fv = remaining_values[feasible_idx]
    fn = len(fw)

    if fn == 1:
        return int(feasible_idx[0])

    k = min(fn, 40)
    densities = fv / fw
    top_k_local = np.argsort(densities)[::-1][:k]

    best_total = -1.0
    best_local = top_k_local[0]

    for i in top_k_local:
        if _get_elapsed() > _DEADLINE - 5.0:
            break
        mask = np.ones(fn, dtype=bool)
        mask[i] = False
        total = fv[i] + _greedy_complete(
            remaining_capacity - fw[i], fw[mask], fv[mask]
        )
        if total > best_total:
            best_total = total
            best_local = i

    return int(feasible_idx[best_local])
# EVOLVE-BLOCK-END
