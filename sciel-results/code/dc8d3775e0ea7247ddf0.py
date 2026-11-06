"""Knapsack heuristic: greedy by value density with feasibility-aware refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0  # hard wall-clock limit from evaluator
_SAFETY = 5.0      # safety margin in seconds


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item index (into the feasible arrays) or -1 to stop.

    Strategy:
      1. If no items or no capacity, stop.
      2. Compute value-density scores for all feasible items.
      3. Use a refined scoring that accounts for:
         - value density (value / weight)
         - a penalty for items that are too heavy relative to remaining capacity
         - a small bonus for items that can still leave room for high-density items
      4. Time-bounded: if computation takes too long, fall back to simple density.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE - _SAFETY

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)

    # Filter to feasible items (weight <= remaining_capacity)
    feasible_mask = w <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    idx_f = np.where(feasible_mask)[0]

    n_f = len(w_f)
    if n_f == 0:
        return -1

    # Base score: value density
    density = v_f / w_f

    # If only one feasible item, take it
    if n_f == 1:
        return int(idx_f[0])

    # Refined scoring: combine density with a capacity-fraction penalty
    # Items that consume a large fraction of remaining capacity get a slight
    # penalty unless they have very high density.
    cap_frac = w_f / remaining_capacity  # fraction of capacity this item uses

    # Score = density * (1 - alpha * cap_frac^2)
    # This penalizes very heavy items quadratically but keeps high-density
    # items preferred.
    alpha = 0.3
    score = density * (1.0 - alpha * cap_frac * cap_frac)

    # Additional refinement: if time permits, do a local lookahead
    # For the top-K candidates by score, simulate adding each and check
    # the marginal value of the best subsequent item.
    if time.time() < deadline and n_f <= 500:
        # Take top-K by base score
        k = min(n_f, 20)
        top_k = np.argsort(-score)[:k]

        best_idx = top_k[0]
        best_val = score[best_idx]

        for ci in top_k:
            if time.time() >= deadline:
                break
            w_after = remaining_capacity - w_f[ci]
            if w_after <= 0:
                # This item fills capacity; its total contribution is just v_f[ci]
                # Compare to current best's density-based score
                continue
            # Best density among remaining items that fit
            w_rem = w_f
            v_rem = v_f
            mask_rem = w_rem <= w_after
            if np.any(mask_rem):
                # Exclude the candidate itself
                mask_rem[ci] = False
                if np.any(mask_rem):
                    next_density = np.max(v_rem[mask_rem] / w_rem[mask_rem])
                    # Combined score: value of this item + expected next
                    # Weight the next item's contribution by how much capacity remains
                    expected_next = next_density * w_after * 0.5  # discount
                    combined = v_f[ci] + expected_next
                    # Normalize to be comparable: divide by total weight used
                    total_w = w_f[ci] + w_after * 0.5
                    if total_w > 0:
                        combined_score = combined / total_w
                        if combined_score > best_val:
                            best_val = combined_score
                            best_idx = ci

        return int(idx_f[best_idx])

    # Fallback: simple density-based pick
    best = int(np.argmax(score))
    return int(idx_f[best])
# EVOLVE-BLOCK-END
