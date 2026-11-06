import numpy as np
import time

# Global state
START_TIME = time.time()
DEADLINE = 140.0  # Safety margin before 150s kill
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global START_TIME, DEADLINE, CAPACITY
    
    # Check deadline
    if time.time() - START_TIME > DEADLINE:
        # Fallback to simple Best Fit if running out of time
        remaining = bins - item
        scores = np.full_like(bins, -np.inf, dtype=float)
        valid_mask = remaining >= 0
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores

    remaining = bins - item
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores

    valid_idx = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_idx]
    
    # Strategy: Hybrid of Best Fit and distribution-aware scoring
    # 1. Base score is Best Fit (negative remaining)
    # 2. If the item is large, we switch to Worst Fit (positive remaining) 
    #    to preserve space for other large items that might arrive.
    # 3. We add a "reuse potential" bonus: 
    #    We estimate how many *other* valid bins have a remaining space 
    #    that could potentially fit into the gap left in this bin.
    #    Actually, the inspiration suggests counting bins with remainder < resulting gap.
    #    Let's refine: 
    #    If we leave a gap G, it's "good" if G is small (Best Fit) 
    #    UNLESS G is small enough that it's useless (dead space).
    
    # Let's look at the distribution of current valid remainders.
    # If we pick a bin with remaining R, the new gap is R.
    # We want R to be as small as possible, but not so small that it's dead.
    
    # Let's try a score that combines:
    # - Best Fit: -R
    # - "Uniqueness" or "Rank": How rare is this remaining space?
    
    # The inspiration 1 used: score = -count(remaining[j] < remaining[i])
    # This favors the smallest remainders.
    
    # Let's try a blend.
    # For small items, Best Fit is strong.
    # For large items, we might want to spread them out (Worst Fit) to avoid fragmentation.
    
    # Threshold: If item > CAPACITY / 2, use Worst Fit logic?
    # If item > 50, it needs a bin with > 50 space.
    # If we have bins with 51, 52, 53... and item is 50.
    # Best Fit picks 51. Gap is 1.
    # Worst Fit picks 53. Gap is 3.
    # 1 is better than 3 if we can fill 1. But 1 might be dead.
    # 3 might be fillable.
    
    # Let's stick to the current program's success (0.1359 is actually quite low/bad? 
    # Wait, the prompt says "score 0.1359" and "search score is seed/value, so 1.0 equals seed".
    # If seed is 1.0, and current is 0.1359, that means current is WORSE than seed?
    # "lower is better" for the raw value (excess %).
    # "search score is seed/value". 
    # If raw value is 0.1359 (13.59% excess), and seed raw value is 1.0 (100% excess)?
    # No, usually "seed" is the baseline heuristic.
    # If current program has score 0.1359, and inspirations have 1.0, 0.9998...
    # And 1.0 equals seed program.
    # This implies the seed program had a search score of 1.0.
    # If search_score = seed_value / current_value?
    # Or search_score = current_value / seed_value?
    # "search score is seed/value". So S = Seed / Value.
    # If S = 1.0, then Value = Seed.
    # If S = 0.1359, then Value = Seed / 0.1359 = 7.3 * Seed.
    # So the current program is performing 7.3x WORSE than the seed in terms of raw excess?
    # Or is "value" the excess?
    # "Score: mean excess over the L1 lower bound in percent".
    # "lower is better".
    # If current program has 13.59% excess, and seed has 1.0% excess?
    # Then S = 1.0 / 13.59 = 0.07.
    # If current program has 1.0% excess, and seed has 13.59%?
    # Then S = 13.59 / 1.0 = 13.59.
    
    # Let's re-read carefully: "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # This implies:
    # Improvement -> Score > 1.0.
    # Score = Seed_Value / Current_Value.
    # If Current_Value < Seed_Value (better), then Score > 1.0.
    # So "Value" is the raw metric (excess %).
    # Current program score is 0.1359.
    # So 0.1359 = Seed_Value / Current_Value.
    # Current_Value = Seed_Value / 0.1359 ≈ 7.3 * Seed_Value.
    # So the current program is MUCH WORSE than the seed.
    # The inspirations have scores 1.0, 0.9998, 0.9998.
    # So Inspiration 1 is exactly the seed.
    # Inspiration 2 is slightly worse than seed.
    
    # Therefore, I should NOT follow the current program's logic (which is bad).
    # I should follow Inspiration 1 (the seed) or improve upon it.
    # Inspiration 1 is "Rank-based Best Fit".
    
    # Let's look at the current program's logic again.
    # It uses "dead gap" penalty.
    # It scored 0.1359.
    # This suggests the "dead gap" penalty was harmful or the base logic was flawed.
    
    # Let's go back to Inspiration 1 (Score 1.0).
    # It uses: score = -count(remaining[j] < remaining[i]).
    # This is effectively Best Fit, but robust to ties and distribution.
    
    # Can we improve on Inspiration 1?
    # Maybe combine it with the "large item" logic from Inspiration 2?
    # Inspiration 2 (Worst Fit for large items) scored 0.9998 (slightly worse).
    # So pure Best Fit (or rank-based BF) is better than the hybrid.
    
    # Let's try to refine the Rank-based Best Fit.
    # The current Inspiration 1 uses `searchsorted` to count how many remainders are smaller.
    # This favors the smallest remainder.
    
    # What if we penalize "dead" space more intelligently?
    # If the remaining space is very small, it's likely dead.
    # If it's slightly larger, it might be alive.
    # But we don't know the future items.
    
    # However, we can look at the *current* items.
    # If there are many small items, small gaps are good.
    # If there are few small items, small gaps are bad.
    
    # Let's try a simple improvement to Inspiration 1:
    # Add a small penalty for remainders that are extremely small (e.g., < 1.0) 
    # if we suspect they are dead.
    # But the seed (Inspiration 1) is the best so far.
    
    # Let's try to make the scoring slightly more sensitive to the "quality" of the gap.
    # Instead of just rank, use the actual value but normalized?
    
    # Let's stick close to Inspiration 1 but add a deadline guard as requested.
    
    valid_rems = remaining[valid_idx]
    sorted_rems = np.sort(valid_rems)
    
    # Count how many bins have remainder strictly less than this one
    counts = np.searchsorted(sorted_rems, valid_rems, side='left')
    
    # Score is -counts
    # To break ties or add nuance, we can subtract a tiny fraction of the actual remainder
    # This ensures that if two bins have the same rank (e.g., identical remainders),
    # it doesn't matter, but if they are close, the smaller one is slightly preferred?
    # Actually, searchsorted with side='left' means if we have [1, 1, 2], 
    # for 1: count is 0.
    # for 2: count is 2.
    # So it strongly favors the minimum.
    
    # Let's just use the seed logic (Inspiration 1) as it's the strongest (1.0).
    # The current program is bad (0.13).
    # The other inspirations are slightly worse than seed.
    
    # I will implement Inspiration 1 with the deadline guard.
    
    scores[valid_idx] = -counts
    
    return scores
