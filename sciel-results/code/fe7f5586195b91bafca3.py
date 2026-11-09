"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # 5s safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _simulate_total(capacity, weights, values, order, skip_indices):
    """Simulate greedy fill following `order`, skipping items in `skip_indices`. Return total value."""
    if isinstance(skip_indices, (int, np.integer)):
        skip_set = {skip_indices}
    else:
        skip_set = set(skip_indices)
    cap = capacity
    val = 0.0
    for i in order:
        if i in skip_set:
            continue
        w = weights[i]
        if w <= cap:
            cap -= w
            val += values[i]
        if cap <= 0:
            break
    return val


def _simulate_2step(capacity, weights, values, order, first_idx, second_order=None):
    """
    2-step lookahead: pick first_idx, then pick the best second item j,
    then continue greedy. Returns max total value over choice of j.
    If second_order is None, use `order` for second item selection.
    """
    w1 = weights[first_idx]
    if w1 > capacity:
        return -1.0
    cap_after_1 = capacity - w1
    v1 = values[first_idx]
    if cap_after_1 <= 0:
        return v1

    # For each feasible second item j (j != first_idx), simulate greedy after picking both
    best_total = v1  # fallback: just pick first item, no second
    search_order = second_order if second_order is not None else order
    for j in search_order:
        if j == first_idx:
            continue
        w2 = weights[j]
        if w2 > cap_after_1:
            continue
        # Simulate greedy fill after picking first_idx and j
        sim_val = v1 + values[j] + _simulate_total(
            cap_after_1 - w2,
            weights,
            values,
            order,
            (first_idx, j)
        )
        if sim_val > best_total:
            best_total = sim_val

    return best_total


def _simulate_3step(capacity, weights, values, order, first_idx, second_idx, third_order=None):
    """
    3-step lookahead: pick first_idx, second_idx, then pick best third item k,
    then continue greedy. Returns total value.
    """
    w1 = weights[first_idx]
    if w1 > capacity:
        return -1.0
    cap_after_1 = capacity - w1
    v1 = values[first_idx]
    if cap_after_1 <= 0:
        return v1

    w2 = weights[second_idx]
    if w2 > cap_after_1:
        return v1
    cap_after_2 = cap_after_1 - w2
    v2 = v1 + values[second_idx]
    if cap_after_2 <= 0:
        return v2

    # For each feasible third item k (k != first_idx, k != second_idx)
    best_total = v2
    search_order = third_order if third_order is not None else order
    for k in search_order:
        if k == first_idx or k == second_idx:
            continue
        w3 = weights[k]
        if w3 > cap_after_2:
            continue
        sim_val = v2 + values[k] + _simulate_total(
            cap_after_2 - w3,
            weights,
            values,
            order,
            (first_idx, second_idx, k)
        )
        if sim_val > best_total:
            best_total = sim_val

    return best_total


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

    # Phase 1: 1-step lookahead for all feasible items
    candidates = []  # list of (score, idx)
    for i in order:
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue  # infeasible
        sim_val = remaining_values[i] + _simulate_total(
            remaining_capacity - w_i,
            remaining_weights,
            remaining_values,
            order,
            i
        )
        candidates.append((sim_val, i))

        if not _elapsed_ok():
            break

    if not candidates:
        return -1

    # Sort by 1-step score descending
    candidates.sort(key=lambda x: -x[0])

    # Phase 2: 2-step lookahead for top candidates
    # Adapt K based on remaining time and problem size
    elapsed = time.time() - _START_TIME
    remaining_time = _DEADLINE - elapsed
    
    # For n=100, each 2-step eval is ~O(n^2) in worst case but typically O(n)
    # Let's be more aggressive: try up to 5 top candidates for 2-step
    # but limit based on time
    if remaining_time > 5.0:
        K = min(5, len(candidates))
    elif remaining_time > 2.0:
        K = min(3, len(candidates))
    else:
        K = min(1, len(candidates))

    best_score = -1.0
    best_idx = candidates[0][1]
    best_second_idx = -1  # track best second item for potential 3-step

    for rank in range(K):
        if not _elapsed_ok():
            break
        _, idx = candidates[rank]
        w_i = remaining_weights[idx]
        if w_i > remaining_capacity:
            continue

        # 2-step lookahead
        sim_val = _simulate_2step(
            remaining_capacity,
            remaining_weights,
            remaining_values,
            order,
            idx
        )
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx

    # Phase 3: 3-step lookahead for the very top candidate if time permits
    elapsed = time.time() - _START_TIME
    remaining_time = _DEADLINE - elapsed
    
    if remaining_time > 3.0 and best_score > 0:
        # Find the best second item for the top candidate
        top_idx = best_idx
        w1 = remaining_weights[top_idx]
        cap_after_1 = remaining_capacity - w1
        v1 = remaining_values[top_idx]
        
        if cap_after_1 > 0:
            # Find best second item
            best_second_score = -1.0
            best_second_idx = -1
            for j in order:
                if j == top_idx:
                    continue
                w2 = remaining_weights[j]
                if w2 > cap_after_1:
                    continue
                sim_val = v1 + remaining_values[j] + _simulate_total(
                    cap_after_1 - w2,
                    remaining_weights,
                    remaining_values,
                    order,
                    (top_idx, j)
                )
                if sim_val > best_second_score:
                    best_second_score = sim_val
                    best_second_idx = j
                if not _elapsed_ok():
                    break
            
            # If we found a good second item, try 3-step
            if best_second_idx >= 0 and not _elapsed_ok() == False:
                sim_3step = _simulate_3step(
                    remaining_capacity,
                    remaining_weights,
                    remaining_values,
                    order,
                    top_idx,
                    best_second_idx
                )
                if sim_3step > best_score:
                    best_score = sim_3step
                    # Note: we still return top_idx as the first pick,
                    # but the 3-step eval confirms it's a good choice

    # If we didn't get to 2-step for the top candidate, use 1-step best
    if best_score < 0:
        best_score, best_idx = candidates[0]

    return int(best_idx)
# EVOLVE-BLOCK-END
