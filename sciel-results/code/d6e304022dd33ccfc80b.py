"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Weibull distribution parameters for the generated sequences
# Assuming a standard Weibull distribution with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and a median of ~19, which fits well into capacity 100.
import scipy.special

k = 2.0
lambda_scale = 25.0
expected_next_item = lambda_scale * scipy.special.gamma(1 + 1/k)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity for the current item
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Thermodynamic Gap Optimization with Probabilistic Penalty
    # We model the bin packing as a system where we want to minimize the "free energy"
    # of the gap distribution. The ideal gap size is derived from the Weibull distribution.
    # We score each bin by the negative of a combined cost function:
    # 1. A term that penalizes gaps that are too small to fit the next item (fragmentation).
    # 2. A term that rewards gaps that are close to the "optimal" gap size (efficiency).
    
    gap = bins - item
    
    # The "optimal" gap size is roughly the expected size of the next item.
    # If the gap is exactly the expected next item size, we are in a stable equilibrium.
    # We use a smooth potential (Gaussian-like) to drive the gap towards this target.
    # However, we also want to avoid large gaps (Worst Fit) if possible, so we combine
    # this with a Best Fit term.
    
    # Let's refine the score from Inspiration 1 but add a stronger pull towards the optimal gap.
    # Score = -gap * (1 + exp(-(gap - expected_next_item)/sigma))
    # This was good, but let's try a different formulation:
    # We want to minimize the "distance" from the ideal state.
    # Ideal state: gap == expected_next_item.
    # Cost = (gap - expected_next_item)^2 + alpha * gap
    # The first term drives the gap to the expected next item size.
    # The second term (alpha * gap) is a Best Fit penalty (minimize gap).
    # We want to MINIMIZE this cost, so we return -cost as the score.
    
    alpha = 0.1  # Weight for Best Fit component
    
    # Calculate the cost
    # We use a smooth approximation to avoid numerical issues
    # Cost = (gap - expected_next_item)^2 + alpha * gap
    
    # To make it more robust, we can normalize the gap by the expected next item
    # Let's try a relative error term
    # Relative Error = (gap - expected_next_item) / expected_next_item
    # Cost = (Relative Error)^2 * expected_next_item + alpha * gap
    
    # Let's stick to the Inspiration 1 structure but tune the sigma and add a direct pull.
    # The Inspiration 1 score was: -gap * (1 + exp(-(gap - expected_next_item)/sigma))
    # This effectively says:
    # If gap >> expected_next_item, multiplier ~ 1, score ~ -gap (Best Fit)
    # If gap << expected_next_item, multiplier ~ 2, score ~ -2*gap (Stronger Best Fit)
    # This seems counter-intuitive for "avoiding small gaps".
    
    # Let's re-read Inspiration 1:
    # "We prefer bins where (bins - item) is large enough to likely fit the next item."
    # "Score = -gap + penalty(gap < expected_next_item)"
    # The code used: scores = -gap * multiplier
    # If gap is small (bad), multiplier is large, so -gap * large is more negative (worse).
    # If gap is large (good), multiplier is small, so -gap * small is less negative (better).
    # Wait, if gap is large, -gap is a large negative number.
    # If multiplier is 1, score is -gap.
    # If gap is small, -gap is a small negative number.
    # If multiplier is 2, score is -2*gap.
    # Comparing -gap (large gap) vs -2*gap (small gap):
    # Example: gap=50, score=-50. gap=5, score=-10.
    # -10 > -50, so the small gap is preferred? That's Best Fit.
    # But we want to AVOID small gaps that can't fit the next item.
    # So if gap < expected_next_item, we should penalize it heavily.
    
    # Let's try a different approach:
    # Score = - (gap + penalty)
    # penalty = max(0, expected_next_item - gap)^2 / sigma
    # This adds a quadratic penalty for gaps that are too small.
    # For gaps that are large enough, penalty is 0, and we just use Best Fit (-gap).
    # This combines Best Fit with a "no fragmentation" constraint.
    
    sigma = 10.0
    deficit = np.maximum(0, expected_next_item - gap)
    penalty = (deficit ** 2) / sigma
    
    scores = -gap - penalty
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
