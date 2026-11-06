"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Online bin packing: `bins` holds the REMAINING capacity of each open bin
# (capacity = 100). For each arriving item we score every bin that can fit it
# and the item is placed in the highest-scoring bin.
#
# Design (kept within the best-fit family, refined):
#   Primary objective: minimize the leftover space after placement
#       leftover = bins - item  (smaller is better)
#   Secondary objective: consolidate into bins that are already well filled,
#       which reduces fragmentation across the many small items produced by
#       Weibull distributions.
#
# We use a SMOOTH surrogate so the score is a well-behaved function of the
# leftover (no nonsmooth min/max):
#   score = -leftover                      (tight fit, dominant)
#           + w * (1 - (leftover/100)^2)   (smooth concave bonus -> fuller bins)
#           + w2 * (item/100)              (tiny constant per item, harmless)
#
# The quadratic term is smooth and concave in leftover, so it is a smooth
# surrogate that gently breaks ties toward fuller bins without ever overriding
# the dominant tight-fit objective (its magnitude is bounded by w <= leftover
# for all realistic leftovers). This corrects the previous version, which
# mislabeled remaining capacity as a fill ratio and thereby rewarded empty bins.

CAP = 100.0
_W = 0.35      # weight of the smooth consolidation bonus
_W2 = 0.05     # tiny item-size normalization term

# Safety margin: the evaluator kills after 150 s; we guard well inside it.
_DEADLINE_MARGIN = 140.0
_t0 = time.time()


def _deadline_exceeded() -> bool:
    return (time.time() - _t0) > _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we are already past the safe budget, fall back to the
    # cheapest correct answer (pure best-fit) so we always return valid scores.
    if _deadline_exceeded():
        remaining = bins - item
        valid = remaining >= 0.0
        scores = np.full_like(remaining, -np.inf, dtype=np.float64)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0.0

    if not np.any(valid):
        # No bin can fit the item; return -inf so the caller opens a new bin.
        return np.full_like(remaining, -np.inf, dtype=np.float64)

    # Smooth surrogate score over all bins (computed vectorised, very fast).
    # Tight-fit term dominates; smooth concave bonus prefers fuller bins.
    scores = -remaining + _W * (1.0 - (remaining / CAP) ** 2) + _W2 * (item / CAP)

    # Invalidate bins that cannot hold the item.
    scores = np.where(valid, scores, -np.inf)

    return scores
# EVOLVE-BLOCK-END
