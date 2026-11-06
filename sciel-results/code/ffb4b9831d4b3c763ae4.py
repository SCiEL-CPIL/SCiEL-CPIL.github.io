"""0/1 Knapsack heuristic: value-function-guided selection with quantized DP and marginal gain scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0  # safety margin before 120s limit


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _build_value_function(weights, values, capacity, n_bands=200):
    """
    Build a quantized value function V(cap) using DP over capacity bands.
    Returns:
        cap_grid: 1D array of capacity breakpoints (length n_bands+1)
        V: 1D array of optimal values at each breakpoint (length n_bands+1)
    """
    if capacity <= 0 or len(weights) == 0:
        return np.array([0.0]), np.array([0.0])

    n = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)

    # Filter to feasible items
    feasible = w <= capacity
    if not np.any(feasible):
        return np.array([0.0]), np.array([0.0])

    w = w[feasible]
    v = v[feasible]
    n = len(w)

    # Quantize capacity into n_bands
    cap_grid = np.linspace(0.0, capacity, n_bands + 1)

    # DP: V[k] = max value achievable with capacity cap_grid[k]
    # Use 1D DP over items, processing capacity bands from high to low
    V = np.zeros(n_bands + 1, dtype=np.float64)

    for i in range(n):
        wi = w[i]
        vi = v[i]
        # For each capacity band, consider adding item i
        # Find the band index where capacity - wi falls
        # V_new[k] = max(V[k], V[band_of(cap_grid[k] - wi)] + vi) if cap_grid[k] >= wi
        # We process bands from high to low to avoid double-counting
        for k in range(n_bands, 0, -1):
            cap_k = cap_grid[k]
            if cap_k < wi:
                continue
            # Find the band for cap_k - wi
            # Since cap_grid is linear, band index ≈ k * (cap_k - wi) / cap_k * n_bands
            # More precisely: find largest j such that cap_grid[j] <= cap_k - wi
            rem_cap = cap_k - wi
            j = int(np.floor(rem_cap / capacity * n_bands))
            j = min(j, n_bands)
            candidate = V[j] + vi
            if candidate > V[k]:
                V[k] = candidate

    return cap_grid, V


def _marginal_gain(item_weight, item_value, cap_grid, V, remaining_cap):
    """
    Estimate marginal gain of adding an item:
    gain = item_value + (V(cap_grid_index(remaining_cap - item_weight)) - V(cap_grid_index(remaining_cap)))
    This uses the value function to estimate how much capacity value we lose.
    """
    if remaining_cap <= 0 or item_weight > remaining_cap:
        return -np.inf

    n_bands = len(cap_grid) - 1
    if n_bands <= 0:
        return item_value

    capacity_total = cap_grid[-1]
    if capacity_total <= 0:
        return item_value

    # Find band index for remaining_cap
    k_cur = int(np.clip(remaining_cap / capacity_total * n_bands, 0, n_bands))
    # Find band index for remaining_cap - item_weight
    rem_after = remaining_cap - item_weight
    k_after = int(np.clip(rem_after / capacity_total * n_bands, 0, n_bands))

    v_cur = V[k_cur]
    v_after = V[k_after]

    # Marginal gain = item value + loss in future value (which is negative if we use capacity)
    # Actually: total with item = item_value + completion_without_item
    # completion_without_item ≈ V at (remaining_cap - item_weight)
    # completion_without_item_baseline ≈ V at remaining_cap (if we didn't pick this item, we'd pick others)
    # Marginal = item_value + V(rem - wi) - V(rem)
    # If this is positive, the item is better than the average alternative
    gain = item_value + v_after - v_cur
    return gain


def _greedy_fill(weights, values, capacity, order=None, exclude_idx=None):
    """
    Greedy 0/1 fill using a given order (or density if order is None).
    Returns (total_value, chosen_mask).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, np.zeros(n, dtype=bool)

    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)

    chosen = np.zeros(n, dtype=bool)
    if exclude_idx is not None:
        chosen[exclude_idx] = False

    if order is None:
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(w > 0, v / w, -np.inf)
        order = np.argsort(-density)

    cap = capacity
    total_val = 0.0
    for idx in order:
        if chosen[idx] or w[idx] > cap:
            continue
        chosen[idx] = True
        total_val += v[idx]
        cap -= w[idx]
        if cap <= 0:
            break

    return total_val, chosen


def _local_refinement(weights, values, capacity, chosen, max_rounds=3):
    """
    Local refinement: 1-swap and insert operations.
    Returns improved total value and chosen mask.
    """
    n = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)

    total_val = float(np.sum(v[chosen]))
    cur_cap = capacity - float(np.sum(w[chosen]))

    for _ in range(max_rounds):
        improved = False
        chosen_pos = np.where(chosen)[0]
        unchosen_pos = np.where(~chosen)[0]

        # Pass A: 1-swap
        for c in chosen_pos:
            wc = w[c]
            vc = v[c]
            best_gain = 0.0
            best_u = -1
            for u in unchosen_pos:
                if chosen[u]:
                    continue
                wu = w[u]
                vu = v[u]
                if wu <= wc + 1e-9:
                    gain = vu - vc
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
            if best_u >= 0:
                total_val += best_gain
                chosen[c] = False
                chosen[best_u] = True
                improved = True
                unchosen_pos = np.where(~chosen)[0]

        # Pass B: Insert
        cur_cap = capacity - float(np.sum(w[chosen]))
        if cur_cap > 0:
            unchosen_pos = np.where(~chosen)[0]
            for u in unchosen_pos:
                if w[u] <= cur_cap + 1e-9:
                    chosen[u] = True
                    total_val += v[u]
                    cur_cap -= w[u]
                    improved = True
                    if cur_cap <= 0:
                        break

        if not improved:
            break

    return total_val, chosen


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using value-function-guided marginal gain scoring.
    """
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

    feasible_mask = w <= cap
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)

    # Build value function over the full capacity range using feasible items
    # Use a moderate number of bands to balance accuracy and speed
    budget = _remaining_budget()
    if budget > 30.0:
        n_bands = 250
    elif budget > 15.0:
        n_bands = 150
    else:
        n_bands = 80

    cap_grid, V = _build_value_function(w, v, cap, n_bands=n_bands)

    # Score each feasible item by marginal gain
    gains = np.full(n, -np.inf, dtype=np.float64)
    for idx in feasible_indices:
        gains[idx] = _marginal_gain(w[idx], v[idx], cap_grid, V, cap)

    # Also compute density as a secondary signal
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(w > 0, v / w, -np.inf)
    density[~feasible_mask] = -np.inf

    # Combined score: primary = marginal gain, secondary = density
    # Normalize gains to [0, 1] range for combination
    gain_min = np.min(gains[feasible_mask])
    gain_max = np.max(gains[feasible_mask])
    gain_range = gain_max - gain_min
    if gain_range < 1e-9:
        normalized_gains = np.zeros(n, dtype=np.float64)
    else:
        normalized_gains = (gains - gain_min) / gain_range
    normalized_gains[~feasible_mask] = -np.inf

    dens_min = np.min(density[feasible_mask])
    dens_max = np.max(density[feasible_mask])
    dens_range = dens_max - dens_min
    if dens_range < 1e-9:
        normalized_dens = np.zeros(n, dtype=np.float64)
    else:
        normalized_dens = (density - dens_min) / dens_range
    normalized_dens[~feasible_mask] = -np.inf

    # Combined score with 80/20 weight on marginal gain vs density
    combined = 0.8 * normalized_gains + 0.2 * normalized_dens
    combined[~feasible_mask] = -np.inf

    # Sort by combined score
    sorted_order = np.argsort(-combined)
    sorted_feasible = sorted_order[feasible_mask[sorted_order]]

    # Adaptive K: number of candidates to evaluate with full lookahead
    if budget > 30.0:
        K = min(n_feasible, 20)
    elif budget > 15.0:
        K = min(n_feasible, 12)
    else:
        K = min(n_feasible, 8)

    best_idx = -1
    best_total = -1.0

    # For each top candidate, do a quick lookahead: greedy fill of the rest
    for i in range(K):
        idx = int(sorted_feasible[i])
        wi = w[idx]
        vi = v[idx]

        rem_cap = cap - wi
        if rem_cap < 0:
            continue

        # Greedy fill of remaining items (excluding idx) using density order
        val_rest, _ = _greedy_fill(w, v, rem_cap, exclude_idx=idx)
        total_est = vi + val_rest

        if total_est > best_total:
            best_total = total_est
            best_idx = idx

        # Time check
        if _remaining_budget() < 1.0:
            break

    # If we have time, do a final local refinement on the best candidate's solution
    if best_idx >= 0 and _remaining_budget() > 3.0:
        wi = w[best_idx]
        vi = v[best_idx]
        rem_cap = cap - wi
        _, chosen_rest = _greedy_fill(w, v, rem_cap, exclude_idx=best_idx)
        # Build full chosen mask
        full_chosen = np.zeros(n, dtype=bool)
        full_chosen[best_idx] = True
        full_chosen[chosen_rest] = True
        # Refine
        _, refined = _local_refinement(w, v, cap, full_chosen.copy(), max_rounds=2)
        refined_val = float(np.sum(v[refined]))
        # If refinement improved, we still pick the same best_idx (the selection is what matters)
        # The refinement is just to validate; the returned index is best_idx
        pass

    return int(best_idx) if best_idx >= 0 else -1
# EVOLVE-BLOCK-END
