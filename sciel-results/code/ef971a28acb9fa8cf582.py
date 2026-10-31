import numpy as np
import time

# Global state to track if we have started timing
_start_time = None

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
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
        
    # Get the remaining capacities of valid bins
    valid_remaining = remaining[valid]
    
    # Strategy: Adaptive Polarization
    # The goal is to steer the remaining capacity of bins towards a "safe zone" 
    # that is neither too small (waste) nor too large (under-utilization for small items).
    # For Weibull distributions with capacity 100, items often cluster in smaller sizes.
    # Keeping bins around 40-60% full (remaining 40-60) is often optimal for mixing.
    
    # Define a target "ideal" remaining capacity.
    # Based on the "Polarization" inspiration, moving away from 50% (half-capacity) 
    # or towards a specific optimal gap can help.
    # Let's try a "Golden Ratio" inspired target for the remaining space, 
    # or simply a mid-point target that adapts to the item size.
    
    # If the item is very small (< 10), we want to pack tightly (Best Fit) to save large bins.
    # If the item is large (> 50), we have fewer choices, Best Fit is usually safe.
    # For medium items, we might want to avoid creating tiny slivers.
    
    # Let's use a dynamic target for remaining capacity.
    # Target remaining = 0.618 * (bins - item) ? No, bins varies.
    # Let's target a specific remaining value R_target.
    # If we place item in bin, new_remaining = bin - item.
    # We want new_remaining to be close to R_target.
    
    # What is a good R_target?
    # If we look at the distribution of items, a target remaining of ~30-40 units 
    # (leaving 60-70% filled) might be good for future small items.
    # However, if we have many large items, we need to keep large bins open.
    
    # Let's analyze the current state of the bins (valid ones).
    # Calculate the mean remaining capacity of valid bins.
    mean_remaining = np.mean(valid_remaining)
    
    # If the average remaining is high, we are in the early phase. 
    # If low, we are in the late phase.
    
    # Heuristic: 
    # Score = -|new_remaining - Target|
    # Where Target depends on the item size and current state.
    
    # Let's try a target that preserves "medium" space.
    # For Weibull (k=2, lambda such that mean is reasonable), 
    # a target remaining of 50 (half-full) is a common baseline.
    # The inspiration "Polarization" suggests maximizing distance from half-capacity 
    # OR adapting. Let's try minimizing distance to a "sweet spot" of 40 units remaining.
    
    # Actually, let's look at the "Polarization" finding: 
    # "maximizing distance from half-capacity ... measured improvement".
    # This implies that bins that are very full or very empty are "better" states to create/maintain 
    # than bins that are exactly half full? Or does it mean we should place items such that 
    # the *resulting* remainder is far from 50?
    
    # Let's test the hypothesis: Score = - (new_remaining - 50)^2 
    # vs 
    # Score = - abs(new_remaining - 50)  (Minimize distance to 50 - Best Fit to 50)
    # vs 
    # Score = abs(new_remaining - 50)    (Maximize distance from 50 - Polarization)
    
    # The prompt says "Polarization scoring (maximizing distance from half-capacity) ... measured improvement".
    # So we should MAXIMIZE the distance of the new remainder from 50.
    # new_remaining = bins[valid] - item
    # Distance from 50 = abs(new_remaining - 50)
    
    # However, we must be careful. If we always maximize distance from 50, 
    # we might fill bins completely (rem=0, dist=50) or leave them empty (rem=100, dist=50).
    # Filling bins completely (rem=0) is generally good (Best Fit behavior for small items).
    # Leaving bins empty (rem=100) is generally bad (Worst Fit behavior, bad for small items).
    
    # So we need a directional polarization. 
    # We prefer small remainders (full bins) over large remainders (empty bins) if possible.
    # But for large items, we are forced to use large bins.
    
    # Let's refine: 
    # If item is small, we want to minimize remaining (Best Fit). 
    # If item is large, we want to minimize the "waste" of using a large bin, 
    # which is also minimizing remaining? 
    # Actually, Best Fit is generally robust. The improvement from Polarization 
    # suggests that *sometimes* we should NOT pick the tightest fit.
    
    # Let's try a hybrid: 
    # Calculate Best Fit score (min remaining).
    # Calculate Polarization score (max distance from 50, but penalize if remaining > 50).
    
    # Let's implement a score that favors remainders in [0, 40] or [60, 100]?
    # No, [60, 100] is bad. We want [0, 40].
    # So we want new_remaining to be small.
    
    # Why would Polarization work then? 
    # Maybe it prevents the "fragmentation" where we have many bins at 49, 48, 51...
    # By pushing remainders to be either very small (<20) or very large (>80), 
    # we create "clean" bins for large items and "full" bins for small items.
    
    # Let's try: Score = -abs(new_remaining - 50) 
    # This maximizes distance from 50.
    # If new_remaining is 0, score = 50.
    # If new_remaining is 100, score = 50.
    # We need to break the tie in favor of 0 (fuller bins).
    
    # Score = -abs(new_remaining - 50) - (new_remaining * 0.01)
    # This prefers 0 over 100.
    
    valid_indices = np.where(valid)[0]
    new_rems = valid_remaining
    
    # Distance from 50
    dist_from_half = np.abs(new_rems - 50.0)
    
    # Base score: Higher is better. We want to maximize dist_from_half.
    # But we also want to prefer lower new_rems if dist is similar.
    # Let's combine:
    # Score = dist_from_half - 0.1 * new_rems
    
    # Let's check:
    # rem=0: dist=50, score=50
    # rem=50: dist=0, score=-5
    # rem=100: dist=50, score=40
    # So rem=0 is preferred over rem=100. Good.
    # rem=20: dist=30, score=28
    # rem=30: dist=20, score=17
    # rem=40: dist=10, score=6
    # rem=60: dist=10, score=-1
    # rem=70: dist=20, score=-2
    # rem=80: dist=30, score=-12
    # rem=90: dist=40, score=-21
    
    # This heavily favors small remainders (0-20) and avoids mid-range (40-60).
    # This aligns with "Polarization".
    
    scores_valid = dist_from_half - 0.1 * new_rems
    
    # Add a tiny index penalty for tie-breaking (lower index preferred)
    scores_valid -= np.arange(len(valid_indices)) * 1e-9
    
    scores[valid] = scores_valid
    
    return scores
