"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: track elapsed time to ensure we finish well within limits.
    # This is a heuristic bin packing priority function.
    
    # Ensure bins is a numpy array for fast operations
    if not isinstance(bins, np.ndarray):
        bins = np.asarray(bins, dtype=np.float64)
    
    # Bins with enough remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # If no valid bins, return all -inf (shouldn't happen in valid evaluation)
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Strategy: Combine best-fit with a "balance" heuristic.
    # Best-fit minimizes remaining space (reduces fragmentation).
    # However, purely best-fit can lead to suboptimal packing in some cases.
    # We use a weighted combination of:
    # 1. Best-fit score: -remaining (prefer smallest remaining)
    # 2. Balance score: prefer bins that are already "busy" but not too full? 
    #    Actually, a common improvement is "Best Fit Decreasing" but we don't know item order.
    #    Another approach: "Worst Fit" is sometimes better for balancing load.
    #    
    # Let's try a hybrid: 
    # Score = -remaining * alpha + (bins / capacity) * beta
    # But we don't know capacity explicitly. Capacity is 100 in the task description.
    # We can assume capacity is the maximum possible bin size, but bins might be initialized to capacity.
    # Actually, the bins array contains the *remaining* capacity or *used* capacity?
    # The task says "bins with enough remaining capacity", and the seed uses `-(bins - item)`.
    # This implies `bins` holds the *remaining* capacity.
    # So if bins[i] is remaining capacity, then used = capacity - bins[i].
    # Capacity is 100.
    
    CAPACITY = 100.0
    
    # For valid bins:
    # remaining[i] is the space left.
    # used[i] = CAPACITY - remaining[i]
    
    # Best fit: minimize remaining -> maximize -remaining
    best_fit_score = -remaining
    
    # Another heuristic: prefer bins that are closer to full? 
    # That's the same as best fit.
    
    # Let's try "Worst Fit" for valid bins: maximize remaining.
    worst_fit_score = remaining
    
    # Hybrid: 
    # If remaining is very small, best fit is good.
    # If remaining is large, worst fit might balance better.
    
    # A simple improvement over best fit: 
    # Use a nonlinear function of remaining.
    # e.g., score = -remaining^2 (prefer bins that are already quite full, but not too much?)
    # Or score = -remaining * (1 + remaining/CAPACITY)
    
    # Let's test a "smoothed best fit" that penalizes large remaining spaces less.
    # Actually, best fit is known to be strong. Let's try a slight variation.
    
    # Idea: Prefer bins where the item fits "snugly" but not too tightly.
    # But too tight can cause future small items to not fit.
    
    # Let's try: score = -remaining + epsilon * (used / CAPACITY)
    # This slightly prefers more used bins (balance).
    
    # Or: score = -remaining * (1 - used/CAPACITY)
    # This reduces the penalty for remaining space in full bins.
    
    # Let's stick to a robust heuristic: 
    # 1. Best fit (minimize remaining)
    # 2. If multiple bins have same remaining, prefer the one with higher used (more balanced?)
    
    # Since we return a single score per bin, we need a scalar.
    
    # Let's try: score = -remaining * (1.0 + 0.1 * (CAPACITY - remaining) / CAPACITY)
    # This makes the penalty for remaining space slightly higher in fuller bins.
    
    # Actually, a very common and effective heuristic is "Best Fit".
    # The seed is Best Fit. To improve, we need something better.
    
    # Consider "First Fit Decreasing" is not applicable since we process online.
    
    # How about "Next Fit"? No, that's worse.
    
    # Let's try a combination of Best Fit and Worst Fit.
    # score = -remaining * 0.7 + remaining * 0.3 = -0.4 * remaining
    # That's just a scaled best fit.
    
    # Let's try: prefer bins that are in the "middle" of fill level?
    # No, that's not standard.
    
    # Research suggests that for online bin packing, Best Fit is often competitive.
    # However, "Harmonic" or other algorithms exist.
    
    # Let's try a simple tweak: 
    # If remaining is very close to 0, it's a perfect fit.
    # If remaining is large, it's a waste.
    
    # Let's use a nonlinear penalty: 
    # score = -np.sqrt(remaining)
    # This reduces the difference between bins with small remaining.
    
    # Or: score = -remaining / (1 + remaining)
    
    # Let's try: score = -remaining * (1 + remaining/CAPACITY)
    # This penalizes large remaining spaces more heavily.
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Compute scores for valid bins
    r = remaining[valid_mask]
    # Use a quadratic penalty to emphasize tight fits
    scores[valid_mask] = -r * (1.0 + r / CAPACITY)
    
    return scores
# EVOLVE-BLOCK-END
