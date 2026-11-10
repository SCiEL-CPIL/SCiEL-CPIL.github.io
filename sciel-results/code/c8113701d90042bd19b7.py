"""Improved 0/1 knapsack heuristic with lookahead evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 110.0  # seconds; safety margin before 120s kill
_START = None


def _greedy_fill_value(weights, values, capacity, exclude_idx=None):
    """Compute total value from greedy density fill, optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    # Create density array
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    # Prefix sums
    cum_w = np.cumsum(w_sorted)
    cum_v = np.cumsum(v_sorted)
    # Find how many full items fit
    # cum_w[idx] <= capacity means items 0..idx fit
    idx = np.searchsorted(cum_w, capacity, side='right') - 1
    if idx < 0:
        # No full item fits; take fraction of first
        return float(v_sorted[0] * (capacity / w_sorted[0])) if w_sorted[0] > 0 else 0.0
    total = float(cum_v[idx])
    remaining = capacity - cum_w[idx]
    if remaining > 0 and idx + 1 < len(w_sorted):
        # Fractional part of next item (for upper bound estimate, but we want integer 0/1)
        # For 0/1, we can't take fraction, so stop here.
        # Actually for lookahead we want the 0/1 greedy value, so no fractional.
        pass
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item for 0/1 knapsack using lookahead evaluation of top-k density candidates.
    """
    global _START
    if _START is None:
        _START = time.time()
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        return int(np.argmax(remaining_values / remaining_weights))

    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1

    # Quick check: if only one item, take it if it fits
    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1

    # Compute densities
    densities = remaining_values / remaining_weights

    # Top-k candidates by density
    k = min(n, 30)
    top_k_idx = np.argsort(densities)[::-1][:k]

    # For each candidate, compute lookahead value:
    # value[i] + greedy_fill(remaining_capacity - weight[i], excluding i)
    best_score = -1.0
    best_idx = top_k_idx[0]

    for i in top_k_idx:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        if w_i > remaining_capacity:
            continue
        rem_cap = remaining_capacity - w_i
        fill_val = _greedy_fill_value(remaining_weights, remaining_values, rem_cap, exclude_idx=int(i))
        total_val = v_i + fill_val
        # Small penalty for wasting capacity (encourages tighter fits)
        waste = rem_cap  # unused capacity after taking i and filling
        # Score: total value, with slight preference for less waste
        score = total_val - 0.001 * waste
        if score > best_score:
            best_score = score
            best_idx = int(i)

    # Check deadline again
    if time.time() - _START > _DEADLINE:
        return int(np.argmax(remaining_values / remaining_weights))

    return best_idx
# EVOLVE-BLOCK-END
