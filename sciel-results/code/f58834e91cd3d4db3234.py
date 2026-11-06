import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Track time to ensure we finish within the 120s limit (though this function is fast)
    # The main cost is in the evaluator's loop, but we must not hang.
    # This function is called O(N) times per sequence. N=1000. 5 sequences.
    # Total calls ~5000. Each call must be very fast.
    
    start_time = time.time()
    deadline = 120.0 - 5.0  # 5s safety margin
    
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
    
    capacity = 100.0
    
    # Strategy: Best-Fit with a smooth tie-breaker and slight bias for "healthier" bins
    # The current program (score 1.0) uses a complex weighted best-fit.
    # Let's analyze the "Best-Fit" family improvements.
    # Pure Best-Fit minimizes `rem`.
    # Tie-breaking is crucial. If two bins have the same `rem`, which one to pick?
    # 1. Pick the one with less used space (Worst-Fit among ties) -> keeps fuller bins for future large items? No, if rem is same, used is same (since capacity is fixed). 
    #    Wait, rem = capacity - used. So if rem is same, used is same.
    #    So ties in rem imply ties in used.
    #    However, floating point precision might cause near-ties.
    #    
    # 2. What if we consider the "impact" of the item on the bin's "flexibility"?
    #    A bin with rem=10 and item=9 leaves 1. Very tight.
    #    A bin with rem=10 and item=1 leaves 9. Loose.
    #    Best-Fit picks the smallest rem regardless of item size.
    #    Is it better to pack small items into bins that are already somewhat full?
    #    Yes, to avoid fragmenting empty bins.
    #    But Best-Fit already does this by preferring bins with small rem (which implies high usage).
    #
    # Let's try a different angle: "Next Fit Decreasing" is offline.
    # Online, "Best Fit" is standard.
    # "Worst Fit" is worse for skewed distributions.
    # "First Fit" is worse.
    #
    # What about "Best Fit Decreasing" approximation?
    # We don't know future items.
    #
    # Let's look at the "smooth surrogate" hint.
    # The objective is to minimize the number of bins.
    # This is equivalent to maximizing the total usage per bin, or minimizing the total remaining capacity of open bins.
    # 
    # Consider the score: S = -rem.
    # This is linear.
    # What if we use a convex penalty on remaining space?
    # S = -rem^2?
    # This penalizes large remainders more heavily.
    # If rem=1, S=-1. If rem=10, S=-100.
    # This makes the preference for small rem even stronger.
    # Does this help?
    # If we have a choice between rem=1 and rem=2, Best-Fit picks 1.
    # With -rem^2, the difference is 1 vs 4, a stronger signal.
    # But the choice is the same.
    # The only difference is in the *relative* ordering if there are other terms.
    #
    # Let's try adding a term that encourages filling bins to a certain "sweet spot"?
    # No, full is best.
    #
    # What if we consider the *future* expected item size?
    # Weibull distribution with shape parameter k.
    # If k < 1, many small items, few large.
    # If k > 1, more uniform.
    # We don't know k.
    #
    # Let's try a simple improvement: 
    # The current program's "weighted best-fit" might be over-complicating.
    # Let's try pure Best-Fit with a very small epsilon bias towards bins that are *less* full?
    # No, that's Worst-Fit.
    #
    # Let's try: Score = -rem + epsilon * (bins / capacity)
    # This is a slight Worst-Fit bias.
    # For Weibull (skewed, many small), Worst-Fit can be bad because it spreads small items, leaving no room for the occasional large item.
    # Best-Fit clusters small items, leaving large bins open.
    # So Best-Fit is generally better.
    #
    # What if we use: Score = -rem / (bins + epsilon)
    # This normalizes the remaining space by the current usage.
    # If a bin is 50% full, rem=50. Score = -50/50 = -1.
    # If a bin is 90% full, rem=10. Score = -10/90 = -0.11.
    # This prefers the 90% full bin (higher score).
    # This is similar to Best-Fit but scaled.
    #
    # Let's try: Score = -rem * (1 + alpha * (bins / capacity))
    # This penalizes remaining space more if the bin is already full.
    # If bin is empty (bins=0), score = -rem.
    # If bin is full (bins=100), score = -rem * (1+alpha).
    # This makes the "cost" of waste higher in full bins.
    # This encourages packing into full bins even more strongly?
    # But Best-Fit already picks the smallest rem, which is in the fullest bins.
    #
    # Let's look at the "inspiration 2" which was a threshold-based heuristic.
    # "Worst-Fit for small items, Best-Fit for large items".
    # This is a known heuristic: "Best Fit with Worst Fit for small items".
    # Why? If an item is very small, putting it in the tightest bin (Best-Fit) might waste a "tight" slot that could have been used for a slightly larger item.
    # Putting it in a loose bin (Worst-Fit) preserves the tight bins.
    #
    # Let's test this hypothesis.
    # Threshold: item < T.
    # If item < T, use Worst-Fit (maximize rem).
    # Else, use Best-Fit (minimize rem).
    #
    # What is T?
    # Maybe T = 0.1 * capacity = 10?
    # Or T = 0.2 * capacity = 20?
    #
    # Let's implement this smooth transition.
    # Instead of a hard switch, use a smooth blend.
    #
    # Score = w * (Best-Fit Score) + (1-w) * (Worst-Fit Score)
    # Best-Fit Score = -rem
    # Worst-Fit Score = rem  (maximize rem)
    #
    # w should be close to 1 for large items, close to 0 for small items.
    # w = sigmoid((item - T) / s)
    #
    # Let's try T = 15, s = 5.
    
    T = 15.0
    s = 5.0
    
    # Smooth weight for Best-Fit
    # w is high (close to 1) when item >> T
    # w is low (close to 0) when item << T
    w = 1.0 / (1.0 + np.exp(-(item - T) / s))
    
    # Best-Fit score: -rem
    # Worst-Fit score: rem
    # We want to maximize the final score.
    # So for Best-Fit, we want high score for small rem. -rem does this.
    # For Worst-Fit, we want high score for large rem. rem does this.
    
    # Combined score:
    # score = w * (-rem) + (1-w) * (rem)
    # score = rem * (1-w - w) = rem * (1 - 2w)
    
    # If w=1 (large item), score = -rem (Best-Fit)
    # If w=0 (small item), score = rem (Worst-Fit)
    
    factor = 1.0 - 2.0 * w
    
    valid_scores = factor * valid_rem
    
    # Add a small tie-breaker to ensure deterministic behavior and slight preference
    # For Best-Fit (w>0.5), factor is negative. We want to maximize score, so we want smaller rem.
    # Tie-breaker: prefer bins with slightly less used space?
    # Let's add a tiny noise or deterministic bias.
    # Using index-based bias for determinism.
    indices = np.arange(len(valid_bins))
    tie_breaker = 0.001 * (1 - indices / max(len(valid_bins), 1))
    
    valid_scores += tie_breaker
    
    scores[valid_mask] = valid_scores
    
    # Check time
    if time.time() > deadline:
         # Return best effort (which is the current scores)
         pass
         
    return scores
