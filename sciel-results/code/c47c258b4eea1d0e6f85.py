"""Rewritten program: 1-step lookahead with greedy fill simulation + deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145  # 5s safety margin before 150s kill
_START_TIME = time.time()

def _greedy_fill(capacity, weights, values):
    """Simulate greedy density-based fill; return total value."""
    if capacity <= 0:
        return 0.0
    mask = weights <= capacity
    if not np.any(mask):
        return 0.0
    w = weights[mask]
    v = values[mask]
    densities = v / w
    order = np.argsort(-densities)
    total_val = 0.0
    rem = capacity
    for idx in order:
        wi = w[idx]
        vi = v[idx]
        if wi <= rem:
            total_val += vi
            rem -= wi
        if rem <= 0:
            break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1

    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: pure density
        weights = np.asarray(remaining_weights)
        values = np.asarray(remaining_values)
        feasible = weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        fi = np.where(feasible)[0]
        d = values[fi] / weights[fi]
        return int(fi[np.argmax(d)])

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = weights[feasible_mask]
    fv = values[feasible_mask]

    n = len(fw)

    # For each candidate, compute: candidate_value + greedy_fill(remaining after candidate)
    # This is 1-step lookahead.
    # To speed up, we can pre-sort by density for the greedy fill.

    # Pre-sort all feasible items by density descending
    densities = fv / fw
    sort_order = np.argsort(-densities)
    sorted_w = fw[sort_order]
    sorted_v = fv[sort_order]

    # For each candidate i, simulate:
    #   take item i, then greedily fill remaining capacity with other items
    # We can do this efficiently by precomputing prefix sums of sorted items,
    # but since we exclude item i, it's a bit more complex.
    # For n <= 100, a simple loop per candidate is O(n^2 log n) total, which is fine.

    best_score = -1.0
    best_candidate = 0

    for i in range(n):
        rem_cap = remaining_capacity - fw[i]
        if rem_cap < 0:
            continue

        # Greedy fill with all items except i
        # We can iterate through sorted items, skipping item i
        total_val = fv[i]
        rem = rem_cap
        for j in range(n):
            sj = sort_order[j]
            if sj == i:
                continue
            if sorted_w[j] <= rem:
                total_val += sorted_v[j]
                rem -= sorted_w[j]
            if rem <= 0:
                break

        if total_val > best_score:
            best_score = total_val
            best_candidate = i

    # Also consider the pure density option as a tiebreaker / sanity check
    # Actually, the lookahead should dominate. Let's also compute pure density score
    # and if the lookahead best is close, prefer higher density for robustness.
    # But let's just go with lookahead.

    return int(feasible_indices[best_candidate])
# EVOLVE-BLOCK-END
