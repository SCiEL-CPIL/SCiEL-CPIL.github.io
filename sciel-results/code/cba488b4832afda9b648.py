"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters calibrated to the high-performing inspiration (score 1.82)
# This assumes the generated sequences have a mean item size around 13-15.
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Probabilistic Best-Fit (PBF)
    
    Structural Change:
    The previous approaches (current and inspirations) relied heavily on the 
    'Dead Space' metric r * S(r), which creates a bimodal preference (small or large gaps).
    This can lead to 'Worst-Fit' behavior for large gaps, potentially opening 
    unnecessary bins or leaving large holes that are hard to fill.
    
    This program shifts the structural priority to **Best-Fit** (minimizing immediate 
    gap) while using the Probabilistic Dead Space metric as a secondary refinement 
    to avoid 'dead zone' gaps (medium-sized gaps that are unlikely to be filled).
    
    The score is a weighted sum:
    Score = - ( alpha * r + (1 - alpha) * NormDeadSpace(r) )
    
    Where:
    - r is the remaining capacity.
    - NormDeadSpace(r) is the normalized dead space metric.
    - alpha is set to 0.8 to strongly favor Best-Fit, ensuring consolidation.
    
    This ensures that for most items, we pack tightly (Best-Fit), but if there is 
    a choice between two bins with very similar small gaps, we prefer the one with 
    lower probabilistic dead space (i.e., the gap is less likely to be 'stuck').
    """
    # Deadline guard
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit
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
    
    # 1. Best-Fit Component: Minimize r
    # We normalize r to [0, 1] relative to the max possible gap (100) to keep scales comparable
    # Or just use raw r, but the Dead Space term needs to be scaled.
    # Let's use a weighted sum of raw values.
    
    # 2. Probabilistic Dead Space Component
    # DeadSpace(r) = r * exp(-(r/lambda)^k)
    # Max value of DeadSpace is at r* = lambda * (1/k)^(1/k).
    # For k=2, lambda=15, r* = 15 / sqrt(2) ~= 10.6.
    # Max DeadSpace ~= 10.6 * exp(-0.5) ~= 6.4.
    # 
    # To blend with Best-Fit (r), we need to scale them.
    # If we use alpha=0.8, the Best-Fit term dominates.
    # We want the Dead Space term to act as a tie-breaker or minor adjustment.
    
    r = np.maximum(valid_rems, 0.0)
    
    # Calculate survival probability S(r)
    ratio = r / _WEIBULL_LAMBDA
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    
    # Dead space metric
    dead_space = r * survival
    
    # Normalize dead space to be comparable to r?
    # r can be up to 100. dead_space is max ~6.
    # If we just sum them, dead_space is negligible.
    # We want dead_space to influence the decision when r is small.
    # Let's scale dead_space up by a factor, or scale r down.
    
    # Structural Choice: 
    # Use a "Penalty" formulation where we minimize:
    # Penalty = r + gamma * dead_space
    # If gamma is small, it's Best-Fit.
    # If gamma is large, it's Probabilistic Fit.
    
    # Let's try gamma = 0.5.
    # For r=1, dead_space ~ 1 * 1 = 1. Penalty = 1 + 0.5 = 1.5.
    # For r=10, dead_space ~ 10 * 0.5 = 5. Penalty = 10 + 2.5 = 12.5.
    # For r=50, dead_space ~ 50 * 0.0001 = 0.005. Penalty = 50 + 0.0025 = 50.
    # This is essentially Best-Fit.
    
    # To make the Probabilistic term matter more for medium gaps, we need to amplify it.
    # Let's use a non-linear blend.
    
    # Alternative Structural Idea: 
    # Score = - ( r * (1 + survival) )
    # This penalizes r more when survival is high (small r).
    # Wait, survival is high for small r.
    # If r is small, survival ~ 1. Score ~ -2r.
    # If r is medium, survival ~ 0.5. Score ~ -1.5r.
    # If r is large, survival ~ 0. Score ~ -r.
    # This effectively *reduces* the penalty for large r compared to small r?
    # No, Best-Fit is -r.
    # This makes large r *less* penalized (higher score) than small r for the same r value?
    # No, the function is monotonic decreasing with r?
    # Let's check derivative of f(r) = r(1+S(r)).
    # f'(r) = 1 + S(r) + r S'(r).
    # S'(r) is negative.
    # For small r, S' is small negative. f' > 0.
    # So f(r) increases with r.
    # So -f(r) decreases with r.
    # This is still Best-Fit, but with a slightly flatter slope for large r.
    # This is a subtle refinement.
    
    # Let's go back to the Weighted Sum with a Tuned Weight.
    # The inspiration 1.82 used: - (dead_space + 0.1 * r).
    # This is Probabilistic-Dominant.
    # I propose: - (0.5 * r + 2.0 * dead_space).
    # This balances them.
    
    # Let's try to be more aggressive on Best-Fit to avoid the "Worst-Fit drift".
    # Weight for r: 1.0
    # Weight for dead_space: 0.5
    # Penalty = r + 0.5 * dead_space
    
    penalty = r + 0.5 * dead_space
    
    scores[valid] = -penalty
    
    return scores
# EVOLVE-BLOCK-END
