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
    
    # Hybrid strategy: 
    # 1. Primary: Best-fit (minimize remaining space) to pack tightly
    # 2. Secondary: Avoid bins that are already very full if there's a slight fit
    #    Actually, for Weibull (often many small items), pure best-fit is strong.
    #    Let's try a slight variation: penalize bins that are "too full" relative to item size?
    #    Or reward bins where the item fits "nicely" (e.g., ratio of item to bin)?
    
    # Let's stick to the Best-Fit family but refine the score.
    # Current: score = -rem
    # Alternative: score = -rem + alpha * (bins / capacity) ? No, capacity is fixed.
    
    # Consider the "Worst-Fit" aspect for very small items? 
    # If item is very small, Best-Fit might fill up bins too fast, leaving medium bins empty.
    # But Best-Fit is generally very robust.
    
    # Let's try a soft-min/soft-max surrogate as suggested by the directive?
    # The objective is minimizing bin count. 
    # In online packing, we can't see the future.
    
    # Let's look at the inspiration 2 (score 0.5552) which was worse.
    # The current program (score 1.0) is Best-Fit.
    
    # Is there a better heuristic within the "fit" family?
    # "First Fit Decreasing" is offline.
    # "Next Fit" is bad.
    # "Best Fit" is good.
    # "Worst Fit" is sometimes better for uniform distributions, but Weibull is skewed.
    
    # Let's try a blend: 
    # Score = -rem + beta * (bins / 100.0)
    # This slightly prefers fuller bins (higher bins value) if the remaining space is similar?
    # No, that would be worst-fit.
    
    # Let's try: Score = -rem / max(rem, epsilon) ? No.
    
    # How about normalizing the remaining space by the bin's current usage?
    # rem_ratio = rem / bins
    # This measures how much of the *remaining potential* is left.
    # If a bin has 90/100 used, rem=10. rem_ratio = 10/90 = 0.11.
    # If a bin has 10/100 used, rem=90. rem_ratio = 90/10 = 9.0.
    # Best fit minimizes rem.
    # Minimizing rem_ratio might be different?
    
    # Let's test the hypothesis: 
    # For Weibull, items are often small. 
    # If we have a bin with 95 used and one with 50 used.
    # Item size 3.
    # Bin 1 rem: 5. Bin 2 rem: 50.
    # Best Fit picks Bin 1.
    # This is usually correct to keep Bin 2 open for larger items.
    
    # What if we add a small penalty for bins that are almost full?
    # No, that's what best fit does implicitly by preferring smaller rem.
    
    # Let's try a different metric: 
    # Minimize the "waste" created.
    # Waste = rem.
    # So Best Fit is minimizing immediate waste.
    
    # Is there a "Lookahead" heuristic?
    # We can't know future items.
    
    # Let's try a slight modification: 
    # If multiple bins have the same remaining space (rare with floats), pick the one with more used space?
    # Or pick the one with less used space (Worst Fit among ties)?
    
    # Given the directive to "investigate improvements within its family" (Best-Fit family)
    # and "consider a smooth surrogate", maybe the current hard selection is suboptimal due to ties or precision?
    # Unlikely to be the main bottleneck.
    
    # Let's try "Best Fit with a bias towards bins that are not yet 'too' full" to preserve some flexibility?
    # Actually, Best Fit *is* the strategy that preserves flexibility for larger items by filling small bins.
    
    # Let's try "Worst Fit" for comparison? No, directive says KEEP this design (Best Fit).
    
    # Let's try a hybrid: 
    # Score = -rem + 0.1 * (bins / 100.0)
    # This slightly prefers fuller bins if rem is very close?
    # No, -rem dominates.
    
    # What if we use: Score = -rem^2 ?
    # This penalizes large remainders more heavily?
    # If rem=5, score=-25. If rem=10, score=-100.
    # This still picks the smallest rem. It just changes the tie-breaking or relative preference if we added other terms.
    
    # Let's try: Score = -rem + 0.01 * bins
    # This is a very slight Worst-Fit bias.
    
    # Let's try: Score = -rem - 0.01 * (bins / 100.0)
    # This is a slight Best-Fit bias (penalizing fuller bins slightly more? No, -0.01*fuller is more negative, so lower score. So it prefers LESS full bins if rem is same).
    
    # Let's stick to the most robust: Pure Best Fit is often optimal for this type of problem.
    # The score 1.0 suggests it's the baseline.
    
    # Let's try to optimize the computation speed?
    # The function is called 1000 times per sequence, 5 sequences. 5000 calls.
    # Numpy overhead might be significant if bins is small?
    # But bins grows.
    
    # Let's try a different heuristic: "Best Fit Decreasing" is offline.
    # "Harmonic" or "MFF" (Most Free First) are other strategies.
    # MFF is Worst Fit.
    
    # Let's try a slight modification to Best Fit:
    # If the item is very small (< 5% of capacity), use Worst Fit?
    # If the item is large, use Best Fit?
    
    # Let's implement this hybrid.
    capacity = 100.0
    threshold = 0.1 * capacity # 10%
    
    if item < threshold:
        # For small items, spread them out to avoid filling bins too quickly with dust?
        # Or pack them tightly to save bins?
        # Usually, packing small items tightly (Best Fit) is better to keep big bins open.
        # But if we have many small items, Best Fit will fill one bin, then the next.
        # Worst Fit would spread them, leaving all bins partially full.
        # For Weibull, if there are large items later, Worst Fit is dangerous.
        # Best Fit is safer.
        
        # Let's try: Best Fit for all, but with a tie-breaker.
        pass
    
    # Let's try a smooth version of Best Fit?
    # The directive mentions "smooth surrogate for a nonsmooth min/max objective".
    # The selection is argmax. The score is linear.
    # Maybe the "nonsmooth" part is the bin capacity constraint?
    
    # Let's try: Score = -exp(rem / scale)
    # This is a smooth penalty.
    # If scale is large, it's linear. If scale is small, it's steep.
    
    # Let's try: Score = -rem * (1 + 0.1 * (bins/100))
    # This penalizes remaining space more in fuller bins?
    # If bin is full (bins=100), penalty is 1.1 * rem.
    # If bin is empty (bins=0), penalty is 1.0 * rem.
    # This makes it prefer emptier bins if rem is the same?
    # Or if rem is slightly larger in an empty bin, it might still pick the fuller bin?
    
    # Let's try the simplest improvement: 
    # Use "Best Fit" but break ties by preferring the bin with *less* used space (Worst Fit among ties).
    # This might help with numerical precision or slight variations.
    
    # Actually, let's look at the score 1.0. It's the seed.
    # Maybe a slight variation like "Best Fit with 1% Worst Fit bias" helps?
    
    valid_scores = -valid_rem + 0.05 * valid_bins / 100.0
    
    scores[valid_mask] = valid_scores
    return scores
