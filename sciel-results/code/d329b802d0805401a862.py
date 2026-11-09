"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity.

    Uses a refined best-fit strategy with a small preference for bins
    that are not too full (to avoid pathological fragmentation), and
    a secondary tie-breaker that favors bins with more remaining capacity
    to preserve flexibility.

    The score is designed so that higher scores are preferred (the item
    goes to the highest-scoring bin).

    We return a vector of scores; bins that cannot hold the item get -inf.
    """
    start = time.perf_counter()
    deadline = start + 115.0  # safety margin

    remaining = bins - item
    feasible = remaining >= 0

    if not np.any(feasible):
        # No feasible bin; return all -inf so no bin is chosen
        return np.full_like(bins, -np.inf, dtype=float)

    # For feasible bins, compute a composite score.
    # We want to prefer bins where remaining is small (best-fit) but
    # also consider the fraction of the bin that will be used.

    # Normalize remaining by bin capacity (bins themselves are capacity)
    # Actually, bins represent current capacity? Let me re-read.
    # "bins: np.ndarray" — in the seed, bins are the remaining capacities.
    # The item goes to a bin if bins[i] >= item. After placement,
    # bins[i] -= item.

    # So bins[i] is the remaining capacity of bin i.
    # We want to minimize bins[i] - item (best fit), i.e., maximize item - (bins[i] - item) = 2*item - bins[i]
    # But that's just best-fit. Let's think more carefully.

    # Best-fit: choose bin minimizing (bins[i] - item) among feasible bins.
    # This is equivalent to maximizing item - (bins[i] - item) = 2*item - bins[i]
    # Or simply maximizing -bins[i] among feasible bins (since item is constant).

    # But we can do better by considering the "waste" in the bin.
    # A common improvement is to consider the ratio or a convex combination.

    # Let's use: score = -(remaining) + epsilon * (item / bins)
    # This slightly prefers bins that are "fuller" relative to their size.

    # Actually, a well-known heuristic is "best-fit with degradation" or
    # considering the next-best bin. Let's try a smooth approach.

    # For each feasible bin i:
    #   primary: -remaining[i]  (best fit)
    #   secondary: prefer larger remaining if primary scores are close

    # Let's compute scores for feasible bins:
    scores = np.full_like(bins, -np.inf, dtype=float)
    feasible_indices = np.where(feasible)[0]
    rem = remaining[feasible_indices]
    bin_vals = bins[feasible_indices]

    # Best-fit score: negative remaining
    primary = -rem

    # Add a small bonus for bins that are not almost empty after placement
    # to preserve flexibility for larger items. This is a heuristic.
    # Use a soft preference: bonus = alpha * (item / bin_vals)
    # This is higher when the item takes a larger fraction of the bin.
    alpha = 0.1
    bonus = alpha * (item / bin_vals)

    scores[feasible_indices] = primary + bonus

    # Check deadline
    if time.perf_counter() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
