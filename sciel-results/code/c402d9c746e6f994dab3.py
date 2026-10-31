"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import expit

# Weibull parameters estimated from typical Weibull sequences with capacity 100
# These are approximate values that can be tuned
WEIBULL_SHAPE = 1.5  # k
WEIBULL_SCALE = 15.0  # lambda

def _weibull_survival_integral(remainder):
    """
    Approximate the integral of the Weibull survival function from remainder to infinity.
    This represents the expected opportunity cost of holding 'remainder' capacity.
    
    For a Weibull distribution with shape k and scale lambda:
    S(x) = exp(-(x/lambda)^k)
    
    The integral ∫_r^∞ S(x) dx can be approximated numerically or via
    the incomplete gamma function. For efficiency, we use a simple
    approximation that captures the key behavior: larger remainders have
    higher expected waste.
    
    A practical approximation: E[waste | remainder=r] ≈ r * S(r)
    This is the expected value of the unused portion given that we have
    r capacity left.
    """
    # Clip to avoid numerical issues
    r = np.maximum(remainder, 0.0)
    
    # Weibull survival function: S(x) = exp(-(x/lambda)^k)
    if r.size == 0:
        return np.array([])
    
    # Compute S(r)
    x = r / WEIBULL_SCALE
    # For large x, (x^k) can overflow, so use log form
    log_xk = WEIBULL_SHAPE * np.log(x + 1e-10)
    S_r = np.exp(-log_xk)
    
    # Approximate integral as r * S(r)
    # This is a reasonable approximation for the expected waste
    integral_approx = r * S_r
    
    return integral_approx

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Stochastic inventory model where the cost of remaining capacity
    is the expected opportunity cost of holding that capacity, approximated
    by the integral of the Weibull survival function.
    
    For each valid bin (one that can hold the item), we compute:
    - remaining_after = bins[i] - item
    - expected_waste = integral of Weibull survival from remaining_after to infinity
    
    The score is -expected_waste, so we prefer bins where the remaining
    capacity has the lowest expected opportunity cost (i.e., where the
    remaining space is most likely to be filled by future items).
    """
    # Track time for deadline guard
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    # Compute remaining capacity after placing the item
    remaining_after = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining_after >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback to simple best-fit if we're running out of time
        valid_remaining = remaining_after[valid_mask]
        scores_valid = -valid_remaining
        scores[valid_mask] = scores_valid
        return scores
    
    # For valid bins, compute the expected waste using Weibull survival integral
    valid_remaining = remaining_after[valid_mask]
    
    # Compute expected waste for each valid bin
    expected_waste = _weibull_survival_integral(valid_remaining)
    
    # Score is negative expected waste: prefer bins with lower expected waste
    scores_valid = -expected_waste
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
