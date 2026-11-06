import numpy as np
import time

# We assume capacity is 100 as per task description
CAPACITY = 100.0

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    # Strategy: "Next-Item Awareness" heuristic
    # We estimate the expected size of the next item from the current bin occupancy distribution.
    # Specifically, we use the mean of the *remaining* spaces of the valid bins as a proxy for the "typical" item size that can fit.
    # However, a better proxy might be the mean of the *items* themselves, but we don't have access to history.
    # Let's look at the distribution of remaining spaces.
    # If we place the current item in bin i, the new remainder is r_i.
    # We want to choose the bin such that r_i is likely to be filled by the next item.
    # If the next item is expected to be size E, then we want r_i to be close to E.
    # But we don't know E.
    # A common heuristic is to assume the next item is similar to the current item or the average item.
    # Let's try estimating the "next item size" as the mean of the *occupied* parts of the bins? No, that's not available directly.
    # Let's try estimating the "next item size" from the distribution of the *remaining* spaces themselves?
    # If the remaining spaces are clustered around a certain value, that might be the "typical" gap.
    # Alternatively, we can use the current item size as a proxy for the next item size (assuming i.i.d. or similar distribution).
    # Let's try scoring based on how well the new remainder matches the current item size.
    # Score_i = - |remaining[i] - item|
    # This favors bins where the remaining space after placing the item is close to the size of the item itself.
    # This is a form of "Next Fit Decreasing" idea but online.
    
    # Let's refine this.
    # If we assume the next item is distributed similarly to the current item, then the probability that the next item fits in the gap r_i is P(X <= r_i).
    # If we assume X is Weibull with some parameters, we could use the CDF. But we don't know the parameters.
    # A simpler heuristic: Minimize the "waste".
    # Waste = r_i - E[next_item].
    # If we estimate E[next_item] ~ item, then we want to minimize |r_i - item|.
    # Let's try this.
    
    valid_rems = remaining[valid_idx]
    
    # Estimate expected next item size.
    # Option 1: Use current item size.
    # Option 2: Use the median of valid remainders?
    # Option 3: Use the mean of valid remainders?
    # Let's try a combination.
    # If we use the current item size, it's a strong heuristic for Weibull distributions where items are not too variable.
    # Let's try: Score_i = - (valid_rems[i] - item)^2
    # This penalizes gaps that are very different from the current item size.
    
    # However, we must ensure that the gap is not too small (waste) or too large (risk of not filling).
    # Actually, if the gap is larger than the item, it might be filled by a smaller item.
    # If the gap is smaller than the item, it will definitely not be filled by an item of size `item`.
    # So, gaps slightly larger than the item are good. Gaps much larger are risky. Gaps smaller are bad.
    # Let's try: Score_i = - max(0, item - valid_rems[i]) - alpha * max(0, valid_rems[i] - item)
    # This penalizes gaps too small heavily, and gaps too large lightly.
    
    # Let's try a simpler approach first: Best Fit Decreasing (BFD) is known to be strong.
    # Online BFD is just Best Fit (BF).
    # The current program is a rank-based BF.
    # Let's try pure BF: Score_i = - valid_rems[i]
    # This picks the bin with the smallest remaining space.
    
    # Let's try a hybrid:
    # If the smallest remainder is very small, use it.
    # If the smallest remainder is large, maybe a different bin is better?
    
    # Let's stick to the "Next-Item Awareness" idea.
    # Estimate next item size as the mean of the *occupied* parts? No.
    # Let's use the current item size as the estimate.
    # Score_i = - |valid_rems[i] - item|
    
    # Let's test this logic.
    # If item=10, and remainders are [5, 10, 20].
    # Scores: -5, 0, -10.
    # Best score is 0 (remainder 10). This is perfect fit for next item of size 10.
    # If remainders are [5, 15, 20].
    # Scores: -5, -5, -10.
    # Tie between 5 and 15.
    # If we pick 5, next item 10 won't fit.
    # If we pick 15, next item 10 fits with 5 left.
    # So we prefer the larger remainder when both are "far" from item?
    # Actually, if remainder < item, it's a dead gap for items >= item.
    # So we should strongly penalize remainder < item.
    
    # Revised Score:
    # If valid_rems[i] < item: Score = - (item - valid_rems[i]) * 2  # Heavy penalty for dead gap
    # Else: Score = - (valid_rems[i] - item)  # Light penalty for excess
    
    # Let's implement this.
    rems = valid_rems
    diffs = rems - item
    
    # Penalty for rems < item (dead space for current item size class)
    # Penalty for rems > item (excess space)
    
    # We want to minimize waste.
    # Waste = max(0, item - rems) + lambda * max(0, rems - item)
    # Let lambda = 0.5 (excess is less bad than dead space)
    
    waste = np.maximum(0, item - rems) + 0.5 * np.maximum(0, rems - item)
    
    # Score is negative waste
    valid_scores = -waste
    
    scores[valid_idx] = valid_scores
    
    return scores
