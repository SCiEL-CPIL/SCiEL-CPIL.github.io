"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Constants for the problem
CAPACITY = 100.0
# Weibull parameters for the generated sequences
WEIBULL_SHAPE = 1.5
WEIBULL_SCALE = 15.0

# Precompute a lookup table for the Weibull hazard rate
# Hazard rate h(x) = f(x) / S(x) for Weibull distribution
# f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
# S(x) = exp(-(x/lambda)^k)
# h(x) = (k/lambda) * (x/lambda)^(k-1)

# We'll create a lookup table for x in [0, CAPACITY]
# The hazard rate tells us the "urgency" of filling a bin to a particular level
# Low hazard at a remainder means it's unlikely to be used soon, so we can afford to leave it
# High hazard means we should try to fill it more

def _compute_hazard_table(shape, scale, capacity, n_points=10000):
    """Precompute hazard rate values for the Weibull distribution."""
    x = np.linspace(0, capacity, n_points)
    x = np.maximum(x, 1e-10)  # Avoid division by zero
    
    # Weibull PDF: f(x) = (k/scale) * (x/scale)^(k-1) * exp(-(x/scale)^k)
    # Weibull CDF: F(x) = 1 - exp(-(x/scale)^k)
    # Weibull Survival: S(x) = exp(-(x/scale)^k)
    # Hazard: h(x) = f(x) / S(x) = (k/scale) * (x/scale)^(k-1)
    
    hazard = (shape / scale) * (x / scale) ** (shape - 1)
    return x, hazard

# Precompute the hazard table at module load time
_x_table, _hazard_table = _compute_hazard_table(WEIBULL_SHAPE, WEIBULL_SCALE, CAPACITY)

# Also precompute the expected next item size based on Weibull distribution
# E[X] for Weibull = scale * Gamma(1 + 1/shape)
from scipy.special import gamma
_EXPECTED_ITEM = WEIBULL_SCALE * gamma(1 + 1.0 / WEIBULL_SHAPE)

# Time tracking for deadline guard
_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on the Weibull hazard rate at the resulting remainder.
    
    Strategy: Model packing as resource scheduling where we minimize "setup changes"
    (new bins) by scoring based on negative hazard at the remainder after placing the item.
    
    - Low hazard remainder: unlikely to be needed soon, safe to leave open
    - High hazard remainder: likely to be needed, prefer to fill more tightly
    
    We combine:
    1. Hazard-based score: prefer remainders with low hazard (less likely to be used)
    2. Best-fit component: prefer smaller remainders to pack tightly
    3. A blend that balances these concerns
    """
    # Check time budget
    if _time_remaining() < 0.05:
        # Emergency: just return best-fit
        remaining = bins - item
        scores = -remaining.copy()
        scores[remaining < 0] = -np.inf
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    feasible_remainders = remaining_after[feasible]
    
    # Look up hazard rates for the feasible remainders
    # Use linear interpolation on the precomputed table
    hazard_values = np.interp(feasible_remainders, _x_table, _hazard_table)
    
    # Normalize hazard values to [0, 1] for consistent scaling
    h_max = np.max(_hazard_table)
    h_norm = hazard_values / h_max
    
    # Score components:
    # 1. Hazard-based: lower hazard at remainder is better (less likely to be needed)
    #    So we want to MINIMIZE hazard, score = -h_norm
    hazard_score = -h_norm
    
    # 2. Best-fit: smaller remainder is better
    #    Normalize remainder to [0, 1]
    rem_norm = feasible_remainders / CAPACITY
    bestfit_score = -rem_norm
    
    # 3. Combined score with adaptive weighting
    # When items are small, best-fit matters more
    # When items are large, hazard matters more (fewer choices)
    item_ratio = item / CAPACITY
    # Weight: as item gets larger, lean more on hazard
    hazard_weight = 0.3 + 0.7 * item_ratio
    bestfit_weight = 1.0 - hazard_weight
    
    combined = hazard_weight * hazard_score + bestfit_weight * bestfit_score
    
    # Additional: penalize remainders that are "too small" to fit the expected next item
    # This avoids creating gaps that are wasted
    expected_next = _EXPECTED_ITEM
    # If remainder < expected_next, it's likely wasted
    # Soft penalty: increase as remainder approaches 0 from below expected_next
    waste_penalty = np.where(
        feasible_remainders < expected_next,
        (expected_next - feasible_remainders) / expected_next,
        0.0
    )
    waste_weight = 0.2
    
    final_scores = combined - waste_weight * waste_penalty
    
    scores[feasible] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
