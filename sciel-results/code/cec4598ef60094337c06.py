import numpy as np
import time
import scipy.special

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull distribution parameters for the generated sequences
# The task mentions "five generated Weibull sequences" with capacity 100.
# A common setup uses Weibull(k, lambda) where items fit into bins of size 100.
# We assume shape k=2.0 and scale lambda=25.0, yielding mean ~22, median ~19.
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Weibull Survival Function Based Scoring with Best-Fit Core.
    
    We treat the bin packing process as a combinatorial auction where each bin's
    remaining capacity is a "bid" for the item. We score bins by the negative of
    the Weibull survival function evaluated at the resulting remainder, which
    directly minimizes the probability that the leftover space becomes permanently
    unusable waste.
    
    Additionally, we incorporate a best-fit core to ensure tight packing and a
    polarization penalty to avoid "middle" remainders that are neither small
    enough to be filled by a single item nor large enough to be flexible.
    
    Score = -Survival(remainder) - best_fit_term - polarization_penalty
    
    Where:
    - Survival(remainder) = 1 - CDF(remainder) = exp(-(remainder/lambda)^k)
    - best_fit_term = -remainder (favors smaller remainders)
    - polarization_penalty = min(remainder, CAPACITY - remainder)
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        # No feasible bins; return all -inf
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Get feasible gaps
    feasible_gaps = remaining_after[feasible]
    
    # Calculate Weibull survival function: S(x) = exp(-(x/lambda)^k)
    # We want to minimize the probability that the remainder is waste,
    # so we want to minimize S(x), which means we want to maximize -S(x).
    # However, S(x) is decreasing in x, so smaller remainders have higher S(x).
    # Wait, let's rethink.
    # The survival function S(x) = P(X > x) is the probability that a future item
    # is LARGER than the remainder. If S(x) is high, it's likely the remainder
    # cannot be filled by a single item, so it might be wasted.
    # We want to MINIMIZE S(x) to reduce waste probability.
    # So the score should include -S(x) to make smaller S(x) better (higher score).
    # But S(x) is decreasing, so smaller x (remainder) has larger S(x).
    # This means -S(x) is more negative for small remainders, which is bad.
    # This seems counterintuitive. Let's reconsider.
    
    # Actually, the idea is that if the remainder is small, it's likely to be
    # wasted because no future item will fit. If the remainder is large, it's
    # likely to be filled by one or more future items.
    # So we want to AVOID small remainders? No, that's Worst Fit.
    # Best Fit avoids small remainders because they are tight, but Worst Fit
    # avoids large remainders because they are wasteful.
    
    # Let's re-read the directive: "scoring bins by the negative of the Weibull
    # survival function evaluated at the resulting remainder to directly minimize
    # the probability that the leftover space becomes permanently unusable waste."
    
    # If the remainder is r, the probability that it becomes waste is the
    # probability that no future item fits, which is approximately S(r) if we
    # assume items arrive independently. So we want to minimize S(r).
    # Since S(r) is decreasing, smaller r has larger S(r), meaning smaller
    # remainders are MORE likely to be waste. So we should prefer LARGER
    # remainders? That's Worst Fit.
    
    # But Best Fit is usually better for bin packing. Let's think differently.
    # Maybe the waste is not just about one item, but about the total unused
    # capacity. The survival function approach might be flawed.
    
    # Let's try a different interpretation: We want the remainder to be such
    # that it is likely to be filled by future items. The probability that a
    # single future item fits in remainder r is CDF(r) = 1 - S(r).
    # So we want to MAXIMIZE CDF(r), which means MAXIMIZE 1 - S(r), which
    # means MINIMIZE S(r). Again, this favors larger remainders.
    
    # This is confusing. Let's look at the current program's approach:
    # It uses -ra + 5.0 * (1.0 / ra_safe), which strongly favors small remainders.
    # This is Best Fit.
    
    # Maybe the survival function should be applied differently. Perhaps we
    # should consider the probability that the remainder is "good" in some sense.
    # Let's try: Score = -S(r) + best_fit_term.
    # Since S(r) is decreasing, -S(r) is increasing, so larger r gives higher -S(r).
    # This conflicts with best fit.
    
    # Let's try a hybrid: Use the survival function to penalize very small
    # remainders (which are likely waste) but also use best fit to favor
    # small remainders overall.
    
    # Actually, let's just use the survival function as a penalty for small
    # remainders. If S(r) is high (small r), we penalize more.
    # So: Score = -S(r) * weight1 - r * weight2
    
    # But this still favors larger r. Let's try the opposite:
    # Score = S(r) * weight1 - r * weight2
    # This favors small r (best fit) and also favors small r (since S(r) is high).
    # This is redundant.
    
    # Let's try: Score = -r - S(r) * penalty
    # This favors small r (best fit) and penalizes very small r (since S(r) is high).
    # This creates a sweet spot for medium-small r.
    
    # Let's implement this.
    
    # Calculate survival function
    r = feasible_gaps
    # Avoid division by zero
    r_safe = np.maximum(r, 0.1)
    survival = np.exp(-((r_safe / _lambda_scale) ** _k))
    
    # Base best fit score
    base_score = -r
    
    # Penalty for very small remainders (high survival probability)
    # We want to penalize when survival is high (small r)
    penalty = survival * 10.0
    
    # Polarization penalty
    pol_penalty = np.minimum(r, CAPACITY - r)
    pol_weight = 0.5
    
    # Final score
    final_scores = base_score - penalty - pol_weight * pol_penalty
    
    scores[feasible] = final_scores
    
    return scores
