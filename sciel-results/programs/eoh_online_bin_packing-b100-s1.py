import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Valid bins have non-negative remaining capacity
    valid_mask = rem >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_rem = rem[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Strategy: Fragmentation Minimization via Remainder Variance
    # The goal is to minimize the number of bins used. A key insight in bin packing
    # is that "fragmentation" (many small, unusable gaps) is detrimental.
    # We model the bin remainder distribution and prefer placements that keep the
    # remainders "tight" (low variance) or aligned with the mean, rather than
    # creating a wide spread of gaps that are too large for small items but too
    # small for large ones.
    
    capacity = 100.0
    
    # 1. Base Score: Best Fit
    # Primary objective is to minimize immediate waste (remaining space).
    # This is the strongest baseline for skewed distributions like Weibull.
    base_score = -valid_rem
    
    # 2. Fragmentation Penalty
    # Calculate the variance of the current valid remainders.
    # If the variance is high, the bins are in a "chaotic" state of gaps.
    # We want to penalize bins that contribute to this chaos.
    # Specifically, we penalize bins whose remainder is far from the mean remainder.
    # This encourages a "cluster" of similar-sized gaps, which are more likely
    # to be filled by future items of similar size (common in Weibull tails).
    
    # Handle edge case where there is only one valid bin (variance is 0)
    n_valid = len(valid_rem)
    if n_valid > 1:
        mean_rem = np.mean(valid_rem)
        # Variance of the remainders
        var_rem = np.var(valid_rem)
        
        # Scale the variance penalty. 
        # A larger variance means the current state is fragmented.
        # We use a smooth surrogate: penalty is proportional to the squared deviation
        # from the mean, weighted by the overall variance level.
        # This effectively says: "If the system is already fragmented (high var),
        # be stricter about picking a bin that fits the 'norm' (mean)."
        
        # Normalize deviation
        deviation = valid_rem - mean_rem
        
        # Weight by variance to make it adaptive. 
        # If var is 0, no penalty. If var is high, deviation matters more.
        # We use a soft-max style weighting or simple linear scaling.
        # Let's use a simple scaled penalty: alpha * (deviation^2) / (var + epsilon)
        # This normalizes the penalty so it's comparable across different states.
        
        alpha = 0.05  # Small weight to not override Best Fit too aggressively
        epsilon = 1e-10
        
        frag_penalty = alpha * (deviation ** 2) / (var_rem + epsilon)
        
        # Subtract penalty from base score (lower score is worse, so we subtract)
        # Wait, higher score is better. Base score is -rem (higher is better, i.e., smaller rem).
        # We want to DISCOURAGE bins that are outliers.
        # So we subtract the penalty.
        valid_scores = base_score - frag_penalty
    else:
        valid_scores = base_score.copy()
    
    # 3. Tie-breaker / Stability
    # If scores are very close, prefer bins that are less full? 
    # Or more full? 
    # Best Fit already prefers fuller bins (smaller rem).
    # Let's add a tiny bias towards bins with more used space to ensure determinism
    # and slightly favor "closing" bins.
    # Actually, pure Best Fit is usually sufficient. 
    # Let's add a very small term to break ties in favor of bins that are 
    # "more standard" (closer to the median usage?) 
    # For now, keep it simple.
    
    scores[valid_mask] = valid_scores
    
    return scores
