"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    Uses a rank-based scoring approach:
    - Score each valid bin by the negative count of other valid bins whose
      remainder after placing the item is smaller than this bin's remainder.
    - This favors bins that leave a remainder in the lower tail of the
      remainder distribution, which helps future items fit more precisely.
    - Ties are broken by preferring smaller remainders (best-fit).
    """
    start_time = time.time()
    
    rem = bins - item
    scores = np.full_like(rem, -np.inf, dtype=np.float64)
    valid = rem >= 0
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    n_valid = len(valid_rem)
    
    if n_valid == 1:
        scores[valid] = 0.0
        return scores
    
    # Rank-based scoring: for each valid bin, count how many other valid bins
    # have a smaller remainder. Score = -count (so fewer smaller remainders = higher score)
    # Actually, we want to MAXIMIZE the likelihood that residual space can be filled.
    # The directive says: "score each candidate bin by the negative count of other bins 
    # whose remainder is smaller than the resulting gap"
    # So score = - (number of valid bins with remainder < this bin's remainder)
    # This means bins with the smallest remainder get score 0 (or close), and bins with
    # large remainder get very negative scores.
    # Wait, that would favor the SMALLEST remainder, which is best-fit.
    
    # Let me re-read: "maximizing the likelihood that the residual space can be exactly 
    # filled by future items to reduce the L1 lower bound excess"
    # If we pick a bin with a small remainder, the leftover gap is small, making it harder
    # to fill. If we pick a bin with a larger remainder, the gap is larger, which might
    # be easier to fill with future items? No, that doesn't make sense either.
    
    # Actually, the insight is: if we have many bins with similar remainders, picking one
    # that creates a remainder that is common in the distribution means future items are
    # more likely to fit into it. So we want to pick a bin whose resulting remainder is
    # in a "dense" part of the remainder distribution.
    
    # The rank-based approach: count how many bins have remainder <= this remainder.
    # Bins whose remainder is in the middle of the distribution (not extreme) might be best.
    
    # Let me just implement what the directive says literally:
    # "score each candidate bin by the negative count of other bins whose remainder 
    #  is smaller than the resulting gap"
    
    # For each valid bin i, count = number of valid bins j (j != i) where rem[j] < rem[i]
    # score[i] = -count
    
    # This means the bin with the smallest remainder gets score 0 (no other bins have smaller rem)
    # The bin with the largest remainder gets score -(n_valid-1)
    # So we're favoring the SMALLEST remainder, which is pure best-fit.
    
    # But the measured improvement was from 0.9998 to 1.0, which is a small improvement.
    # Let me think about this differently.
    
    # Actually, for Weibull distributions, items tend to be small. So best-fit might be good.
    # But the fragmentation issue is real: if you always pick the tightest fit, you create
    # tiny gaps that can never be filled.
    
    # Let me try a hybrid: use the rank-based score but add a small penalty for very small
    # remainders (relative to the item size) to avoid creating unusable fragments.
    
    # Compute ranks: for each valid bin, how many valid bins have strictly smaller remainder
    # Using argsort for efficiency
    sorted_rem = np.sort(valid_rem)
    
    # For each element, count how many are smaller than it
    # np.searchsorted gives the index where the element would be inserted
    # The number of elements strictly smaller is the insertion point
    counts = np.searchsorted(sorted_rem, valid_rem, side='left')
    
    # scores = -counts would favor smallest remainder
    # But let's add a fragmentation-aware adjustment
    
    # If the remainder is very small compared to the item, it's likely unusable
    # Add a penalty for small remainders
    if item > 1e-9:
        ratio = valid_rem / item
        # Penalty: if ratio is very small, add a negative bonus (lower score)
        # Use a smooth function: penalty = -alpha * exp(-beta * ratio)
        # When ratio is small, exp(-beta*ratio) is close to 1, so penalty is -alpha
        # When ratio is large, exp(-beta*ratio) is close to 0, so penalty is ~0
        alpha = 0.5
        beta = 2.0
        frag_penalty = -alpha * np.exp(-beta * ratio)
    else:
        frag_penalty = np.zeros_like(valid_rem)
    
    # Combine: base score from rank (favoring smaller remainders) + fragmentation penalty
    base_score = -counts.astype(np.float64)
    combined = base_score + frag_penalty
    
    scores[valid] = combined
    return scores
# EVOLVE-BLOCK-END
