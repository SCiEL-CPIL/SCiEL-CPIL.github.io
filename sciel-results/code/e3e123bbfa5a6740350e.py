"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using a composite score blending value density
    with a small preference for lighter items to preserve capacity.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Time budget guard
    if time.monotonic() - _START > _TOTAL_BUDGET:
        return -1

    if remaining_capacity <= 0:
        return -1

    # Only consider items that actually fit
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]

    # Calculate value density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = feasible_values / feasible_weights
    densities = np.where(np.isinf(densities), 1e9, densities)
    densities = np.where(np.isnan(densities), 0.0, densities)

    # Composite score: density with small penalty for heavier items
    # This slightly favors lighter items among similar densities,
    # preserving capacity for more items overall
    alpha = 0.05  # small weight penalty coefficient
    weight_penalty = alpha * feasible_weights / remaining_capacity
    weight_penalty = np.clip(weight_penalty, 0.0, 1.0)
    composite_score = densities * (1.0 - weight_penalty)

    # Find the best candidate
    best_local_idx = int(np.argmax(composite_score))
    best_global_idx = int(feasible_indices[best_local_idx])

    # If the best item's score is very close to the second best,
    # prefer the lighter one to maximize flexibility
    if n > 1:
        sorted_scores = np.sort(composite_score)[::-1]
        if sorted_scores[1] > 0:
            ratio = sorted_scores[0] / sorted_scores[1]
            if ratio < 1.02:  # within 2% of each other
                # Among top candidates (within 2%), pick the lightest
                threshold = sorted_scores[0] * 0.98
                top_candidates = np.where(composite_score >= threshold)[0]
                if len(top_candidates) > 1:
                    lightest_local = top_candidates[np.argmin(feasible_weights[top_candidates])]
                    best_global_idx = int(feasible_indices[lightest_local])

    return best_global_idx
# EVOLVE-BLOCK-END
