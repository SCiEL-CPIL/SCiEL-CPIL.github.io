"""0/1 Knapsack heuristic: lookahead with greedy+swap completion and top-K candidate pruning."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 112.0  # safety margin before 120s limit


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _completion_value(remaining_capacity, weights, values, exclude_idx=None):
    """
    Estimate achievable 0/1 value for remaining items:
    1) Greedy density fill (0/1)
    2) One pass of 1-swap local refinement
    Optionally exclude one item.
    """
    if remaining_capacity <= 0 or len(weights) == 0:
        return 0.0

    n = len(weights)
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
    else:
        mask = np.ones(n, dtype=bool)

    mask &= (weights <= remaining_capacity)
    if not np.any(mask):
        return 0.0

    idxs = np.where(mask)[0]
    fw = weights[idxs]
    fv = values[idxs]
    m = len(fw)
    if m == 0:
        return 0.0

    # Sort by density descending
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / fw, np.inf)
    order = np.argsort(-dens)

    # Greedy 0/1 fill
    selected = []
    total_val = 0.0
    cur_cap = remaining_capacity
    for i in order:
        w = fw[i]
        v = fv[i]
        if w <= cur_cap:
            selected.append(i)
            total_val += v
            cur_cap -= w
            if cur_cap <= 0:
                break

    if len(selected) == 0:
        return 0.0

    # 1-swap refinement
    sel_set = set(selected)
    unselected = [i for i in order if i not in sel_set]
    if not unselected:
        return total_val

    sel_arr = np.array(selected)
    sel_w = fw[sel_arr]
    sel_v = fv[sel_arr]
    unsel_arr = np.array(unselected)
    unsel_w = fw[unsel_arr]
    unsel_v = fv[unsel_arr]

    best_gain = 0.0
    best_swap = None

    for si in range(len(sel_arr)):
        sw = sel_w[si]
        sv = sel_v[si]
        cur_total_w = remaining_capacity - cur_cap
        feasible_unsel = unsel_w <= (cur_cap + sw)
        if not np.any(feasible_unsel):
            continue
        gains = unsel_v[feasible_unsel] - sv
        if np.any(gains > 0):
            best_in_group = np.argmax(gains)
            feasible_positions = np.where(feasible_unsel)[0]
            pos = feasible_positions[best_in_group]
            gain = gains[best_in_group]
            if gain > best_gain:
                best_gain = gain
                best_swap = (si, pos)

    if best_swap is not None:
        si, ui = best_swap
        total_val += best_gain

    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using lookahead: for each top-K candidate by density,
    estimate total value = candidate value + LP relaxation of remaining capacity.
    """
    # Deadline guard
    if _remaining_budget() < 0.5:
        # Fallback: simple density greedy
        w = np.asarray(remaining_weights, dtype=np.float64)
        v = np.asarray(remaining_values, dtype=np.float64)
        cap = float(remaining_capacity)
        if len(w) == 0 or cap <= 0:
            return -1
        feasible = w <= cap
        if not np.any(feasible):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(w > 0, v / w, -np.inf)
        density[~feasible] = -np.inf
        best = int(np.argmax(density))
        return best if feasible[best] else -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Find feasible items
    feasible_mask = w <= cap
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]

    # Compute density for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w > 0, v / w, -np.inf)

    # Sort feasible indices by density descending
    feasible_densities = densities[feasible_indices]
    sorted_order = np.argsort(-feasible_densities)
    sorted_feasible = feasible_indices[sorted_order]

    # Adaptive K: more candidates when more time available
    budget = _remaining_budget()
    if budget > 30.0:
        K = min(len(sorted_feasible), 25)
    elif budget > 15.0:
        K = min(len(sorted_feasible), 18)
    elif budget > 8.0:
        K = min(len(sorted_feasible), 12)
    else:
        K = min(len(sorted_feasible), 8)

    best_idx = -1
    best_estimated_value = -1.0

    for i in range(K):
        idx = sorted_feasible[i]
        wi = w[idx]
        vi = v[idx]

        rem_cap = cap - wi
        if rem_cap < 0:
            continue

        # Greedy+swap completion of remaining items, excluding this candidate
        completion_val = _completion_value(rem_cap, w, v, exclude_idx=idx)
        total_est = vi + completion_val

        if total_est > best_estimated_value:
            best_estimated_value = total_est
            best_idx = idx

        # Time check inside loop
        if _remaining_budget() < 1.0:
            break

    return int(best_idx) if best_idx >= 0 else -1
# EVOLVE-BLOCK-END
