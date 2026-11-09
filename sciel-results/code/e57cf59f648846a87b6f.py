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
    
    # Fragmentation-aware scoring:
    # The key insight is that moderate-sized remainders (fragments) are the most harmful
    # because they are too small to hold many future items but still waste bin capacity.
    # We want to either:
    # 1. Fill the bin almost completely (small remainder) - good
    # 2. Leave a large remainder that can hold many future items - good
    # 3. Avoid creating moderate remainders (fragments) - bad
    
    # Ratio of item to remaining capacity before placement
    # ratio = item / (rem + item) = item / bin_capacity_before
    # If ratio is close to 1, we're filling the bin (good)
    # If ratio is small, we're preserving space (good)
    # If ratio is moderate (0.2-0.6), we create fragments (bad)
    
    ratio = item / (rem + item + 1e-6)
    
    # Sigmoid-based fragmentation penalty:
    # Smooth transition from "good" (low or high ratio) to "bad" (moderate ratio)
    # Use two sigmoids to create a U-shaped or bimodal preference
    
    # Penalty for moderate ratios (fragmentation zone: ratio in [0.15, 0.65])
    # Sigmoid centered at 0.40, width 0.15
    center_frag = 0.40
    width_frag = 0.12
    frag_penalty = -6.0 / (1.0 + np.exp(-(ratio - center_frag) / width_frag)) + \
                   6.0 / (1.0 + np.exp((ratio - center_frag) / width_frag))
    # This creates a penalty that peaks around center_frag and is small at extremes
    
    # Better: use a Gaussian-like penalty but with sigmoid transitions for smoothness
    # Actually, let's use a different approach: score based on "utility" of the remainder
    
    # Utility model: the remainder is useful if it can hold at least one more item
    # Expected item size for Weibull(2, scale) with mean ~10: scale ≈ 5.6
    # A remainder is "wasteful" if it's between 5 and 40 (can't hold many, but not zero)
    
    # Smooth utility function:
    # - rem < 5: nearly full, good (high score)
    # - 5 < rem < 40: fragment zone, bad (low score) 
    # - rem > 40: large remainder, good (high score)
    
    # Use smooth step functions
    low_thresh = 8.0
    high_thresh = 35.0
    width_low = 5.0
    width_high = 8.0
    
    # Smooth step: 0 below (thresh - width), 1 above (thresh + width)
    def smooth_step(x, thresh, width):
        z = (x - thresh) / width
        return 1.0 / (1.0 + np.exp(-z))
    
    # Score components:
    # 1. Near-full bonus: high when rem is very small
    near_full = smooth_step(low_thresh - rem, 0, width_low)  # 1 when rem < low_thresh - width_low
    
    # 2. Large remainder bonus: high when rem is large
    large_rem = smooth_step(rem - high_thresh, 0, width_high)  # 1 when rem > high_thresh + width_high
    
    # 3. Fragment penalty: high when rem is in the middle
    # Use a smooth "bump" function that peaks in the fragment zone
    mid_center = (low_thresh + high_thresh) / 2.0
    mid_width = (high_thresh - low_thresh) / 2.0
    # Bump: exp(-((rem - mid_center) / mid_width)^2)
    bump = np.exp(-((rem - mid_center) / mid_width) ** 2)
    
    # Combine: score = near_full + large_rem - bump * penalty_weight
    penalty_weight = 4.0
    base_score = 2.0 * near_full + 2.0 * large_rem - penalty_weight * bump
    
    # Add a small preference for smaller remainders among "good" placements
    # to break ties and slightly favor best-fit when both are "good"
    tie_breaker = -0.1 * np.sqrt(rem + 1e-6)
    
    scores[feasible] = base_score + tie_breaker
    
    if time.time() > deadline:
        return scores
        
    return scores
