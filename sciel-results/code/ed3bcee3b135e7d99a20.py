import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0
    
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    rem = remaining[feasible]
    
    # Combined strategy:
    # 1. Bimodal residual scoring (from current program):
    #    - Prefer placements where the item nearly fills the bin (ratio ~ 1.0)
    #    - OR leaves a large remainder (ratio small, preserving bin for future items)
    #    - Penalize moderate ratios that create unusable fragments
    # 2. Next-Item Compatibility (from directive):
    #    - Estimate expected next item size from Weibull distribution
    #    - Score bins by probability that residual will accommodate next item
    #    - This prioritizes bins that maintain high utility for immediate future
    
    # Estimate expected next item size
    # For Weibull with capacity 100, typical mean is around 10-20
    # Use a conservative estimate of 15.0 as the expected next item size
    expected_next = 15.0
    
    # Calculate ratio: how much of the remaining space the current item uses
    ratio = item / (rem + 1e-6)
    
    # Base score: mild preference for smaller remainders (best-fit component)
    # Use a gentler power to avoid over-penalizing larger remainders
    base = -np.sqrt(rem + 1e-6)
    
    # Strong fragmentation penalty: discourage ratio in [0.15, 0.60]
    # Wider and stronger than before to clearly create the bimodal preference
    center = 0.35
    width = 0.20
    penalty = -8.0 * np.exp(-((ratio - center) ** 2) / (2 * width ** 2))
    
    # Bonus for very high utilization (ratio close to 1.0) - fill the bin
    high_util_bonus = 3.0 * np.exp(-((ratio - 0.95) ** 2) / (2 * 0.05 ** 2))
    
    # Bonus for low utilization (ratio close to 0) - preserve bin space
    # This is the second mode of the bimodal distribution
    low_util_bonus = 2.0 * np.exp(-((ratio - 0.05) ** 2) / (2 * 0.10 ** 2))
    
    # Next-Item Compatibility score:
    # Probability that the remaining space after placing current item
    # will accommodate the expected next item
    # If rem >= expected_next, it's likely to fit; otherwise, it's risky
    # Use a smooth transition around the threshold
    next_compat = np.exp(-np.maximum(0, expected_next - rem) / 5.0)
    next_compat_score = 4.0 * next_compat
    
    # Combine all components
    scores[feasible] = base + penalty + high_util_bonus + low_util_bonus + next_compat_score
    
    if time.time() > deadline:
        return scores
        
    return scores
