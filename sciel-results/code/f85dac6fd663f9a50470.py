"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for estimating fill probability of remainders
# These are tuned for typical Weibull sequences with capacity 100
# k=2.0, lambda=15.0 implies mean ~ 15 * Gamma(1.5) = 15 * 0.886 = 13.3
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Distribution-aware "Expected Dead Space" minimization.
    
    We model the future items as Weibull(k, lambda).
    If we place an item in a bin leaving remainder `r`, we incur "dead space"
    risk. The expected dead space is related to the integral of the survival
    function, but a simpler proxy that captures the trade-off between
    consolidation (small r) and future usability (large r) is the product
    of the remainder and its survival probability:
    
    DeadSpace(r) = r * S(r) = r * exp(-(r/lambda)^k)
    
    This function has a minimum at r=0 (perfect fit) and r->infinity (no waste
    relative to bin size, though we can't have infinity). It has a maximum
    (worst dead space) at some intermediate r*.
    
    By minimizing DeadSpace(r), we encourage:
    1. Small remainders (Best Fit behavior) because r is small.
    2. Very large remainders are less penalized than medium ones because S(r)
       becomes very small, acknowledging that large gaps are "safe" from being
       fragmented by small items, but we still prefer consolidation via the
       r term dominating for moderate r.
    
    Actually, let's look at the derivative of f(r) = r * exp(-(r/lambda)^k).
    f'(r) = exp(-(r/lambda)^k) * (1 - k*r^k/lambda^k).
    f'(r) = 0 when 1 = k*(r/lambda)^k => r = lambda * (1/k)^(1/k).
    For k=2, r* = lambda / sqrt(2) ~ 15 / 1.414 ~ 10.6.
    So remainders around 10 are the "worst" dead space.
    Remainders < 10 are better (closer to 0).
    Remainders > 10 are better (closer to infinity).
    
    This creates a bimodal preference: either pack tight (small r) or leave a
    large gap (large r). This is structurally different from pure Best Fit
    (which always prefers small r) or Worst Fit (which always prefers large r).
    It avoids the "dead zone" of medium-sized gaps that are too small to be
    useful for large items but too large to be ignored, while also not
    forcing Worst Fit which wastes capacity.
    
    Score = - (r * exp(-(r/lambda)^k))
    
    We add a small tie-breaker term to prefer smaller remainders among those
    with similar dead space, or simply rely on the shape.
    """
    global _START_TIME
    
    # Initialize timing if not started
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Compute the "Dead Space" metric: r * S(r)
    # S(r) = exp(-(r/lambda)^k)
    # To avoid overflow/underflow issues and for numerical stability,
    # we compute the score as - (r * exp(-(r/lambda)^k))
    
    # Handle r=0 case: 0 * exp(0) = 0. Score = 0.
    # Handle large r: r * ~0 = ~0. Score = ~0.
    # Handle medium r: max value. Score is most negative.
    
    # Vectorized computation
    # Clip r to non-negative just in case of floating point errors
    r = np.maximum(valid_rems, 0.0)
    
    # Calculate survival probability S(r)
    # (r / lambda)^k
    ratio = r / _WEIBULL_LAMBDA
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    
    # Dead space metric
    dead_space = r * survival
    
    # Score: minimize dead space
    # Higher score is better, so we negate the dead space
    # We might add a small penalty for large remainders to prevent pure Worst Fit
    # if the dead space metric plateaus near 0 for very large r.
    # However, the r term in dead_space = r * S(r) grows linearly with r
    # before S(r) decays. The maximum is at r*.
    # For r > r*, dead_space decreases.
    # So very large r has small dead_space (good score).
    # This effectively encourages Worst Fit for large bins?
    # Let's check the behavior.
    # If r is very large, dead_space is small. Score is high.
    # If r is small, dead_space is small. Score is high.
    # If r is medium, dead_space is large. Score is low.
    
    # This is a "V-shaped" or "U-shaped" score landscape (inverted U for dead space).
    # This might be too aggressive towards Worst Fit.
    # Let's blend it with Best Fit to ensure consolidation.
    # Score = - (dead_space + beta * r)
    # The beta * r term ensures that as r -> infinity, score -> -infinity,
    # preventing pure Worst Fit.
    # This creates a single peak at an optimal remainder size.
    
    beta = 0.1  # Small penalty for large remainders to encourage consolidation
    
    combined_dead_space = dead_space + beta * r
    
    scores[valid] = -combined_dead_space
    
    return scores
# EVOLVE-BLOCK-END
