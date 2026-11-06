"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = 150.0

# Representative future item sizes for Weibull (shape~1.7, scale~14, cap=100)
# These span the likely range of arriving items.
_REP_SIZES = np.array([5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 38.0, 45.0, 55.0])

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Design (kept in same family: best-fit + smooth gap penalty):
    1. Primary: minimize remaining capacity (best-fit)
    2. Secondary: smooth "usable gap" score via log-sum-exp over representative
       future item sizes. This approximates the expected number of future items
       that could be placed in the remaining gap, penalizing gaps that are too
       small to hold a typical item or too large to be efficiently filled.
    """
    elapsed = time.time() - _start_time
    if elapsed > _deadline - 0.5:
        remaining = bins - item
        valid = remaining >= 0
        if not np.any(valid):
            return np.full(len(bins), -np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(valid, -remaining, -np.inf)
        return score

    remaining = bins - item
    valid = remaining >= 0

    if not np.any(valid):
        return np.full(len(bins), -np.inf)

    # Primary component: best-fit (minimize remaining)
    # Use a scaled version so it's comparable to the secondary term
    primary = -remaining

    # Secondary component: smooth "usable gap" via log-sum-exp soft-min
    # For each bin, compute how many of the representative sizes fit in the gap.
    # usable_count[gap] = sum over rep_sizes of indicator(rep_size <= gap)
    # We smooth this: soft_usable = sum over rep_sizes of sigmoid((gap - rep_size) / width)
    # This is a smooth surrogate for the count of fitting items.
    #
    # But we want to PENALIZE gaps that are "wasteful": too small to hold a typical
    # item, or so large that many items could fit (wasting potential).
    # Actually, for best-fit we want SMALL gaps. The penalty should be:
    #   - If gap is very small (< min rep size), it's fine (tight fit).
    #   - If gap is moderate, it's good (one more item could fit).
    #   - If gap is very large, it's bad (wasted space).
    #
    # A clean smooth surrogate: penalty = -log(1 + sum of exp((gap - s_i)/tau))
    # This grows smoothly as gap exceeds the representative sizes.
    #
    # Simpler and effective: use a smooth step that is ~0 for small gaps and
    # grows for large gaps. Use the soft-min formulation:
    #   soft_penalty = -tau * log(sum_i exp((gap - s_i) / tau))
    #   This is a smooth approximation of -min_i(s_i - gap) when gap < s_i,
    #   and grows linearly when gap > max(s_i).
    #
    # Actually, let's use a direct smooth penalty:
    #   penalty = sum_i sigmoid((gap - s_i) / width)
    #   This counts (smoothly) how many representative sizes the gap exceeds.
    #   Small gap -> penalty ~0. Large gap -> penalty ~N.
    #   We want to MINIMIZE this, so we subtract it from the score.

    gap = remaining[valid]  # only valid bins
    # Use a moderate width for smoothness
    width = 5.0
    # Compute smooth count of representative sizes that fit in the gap
    # For each gap value, count how many rep sizes are <= gap (smoothly)
    # gap shape: (n_valid,), _REP_SIZES shape: (10,)
    # diff shape: (n_valid, 10)
    diff = gap[:, None] - _REP_SIZES[None, :]
    # sigmoid(diff / width) is ~1 when gap >> s_i, ~0 when gap << s_i
    smooth_count = np.mean(1.0 / (1.0 + np.exp(-diff / width)), axis=1)
    
    # The penalty: we want to penalize LARGE gaps. 
    # smooth_count is high for large gaps. So penalty = smooth_count.
    # But we also want to REWARD gaps that are "just right" (can hold one more item).
    # Actually, for best-fit, smaller gaps are always better. The smooth_count
    # term just adds a smooth penalty for large gaps, which is what we want.
    
    # Combine: score = primary - weight * penalty
    # primary is negative (we want to maximize, so -remaining is good for small remaining)
    # penalty is positive for large gaps (we want to minimize, so subtract)
    weight = 0.5
    valid_scores = primary[valid] - weight * smooth_count

    # Build full score array
    score = np.full(len(bins), -np.inf)
    score[valid] = valid_scores

    return score
# EVOLVE-BLOCK-END
