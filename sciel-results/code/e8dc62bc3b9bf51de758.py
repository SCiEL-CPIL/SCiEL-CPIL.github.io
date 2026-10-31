"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We precompute nothing heavy; the function must be fast per call.
# Strategy: Polarization scoring.
# We want to push bins towards being either full (remainder ~ 0) or empty (remainder ~ capacity).
# "Half-full" bins are bad because they can't fit large items and waste space if small items don't fill them perfectly.
# 
# Score = -(distance from 0 + distance from capacity) is equivalent to maximizing (min(rem, cap-rem)).
# Wait, if rem=0, min(0, cap)=0. If rem=cap/2, min(cap/2, cap/2)=cap/2.
# So maximizing min(rem, cap-rem) pushes to the middle. That's BAD.
#
# We want to MINIMIZE min(rem, cap-rem)?
# If rem=0, min=0. If rem=cap/2, min=cap/2.
# So minimizing min(rem, cap-rem) pushes to 0 or cap. This is polarization.
#
# Let's check:
# Bin A: rem = 1. min(1, 99) = 1.
# Bin B: rem = 50. min(50, 50) = 50.
# Bin C: rem = 99. min(99, 1) = 1.
# We prefer A and C over B.
# Score should be higher for A and C.
# Score = -min(rem, cap-rem) ?
# A: -1
# B: -50
# C: -1
# Max score is -1. So A or C is chosen. Correct.
#
# However, there is a nuance. If we have a very small item, putting it in a nearly full bin (rem=1) makes it rem=0.5.
# min(0.5, 99.5) = 0.5. Score -0.5.
# Putting it in a half-full bin (rem=50) makes it rem=49.5.
# min(49.5, 50.5) = 49.5. Score -49.5.
# -0.5 > -49.5. So it goes to the nearly full bin. Good.
#
# What if we have a large item?
# Item size 60.
# Bin A: rem=1. Cannot fit.
# Bin B: rem=50. Cannot fit.
# Bin C: rem=99. Fits. rem becomes 39.
# min(39, 61) = 39. Score -39.
# Only C is valid.
#
# What if we have two bins that fit?
# Bin D: rem=60. Item 30. New rem=30. min(30,70)=30. Score -30.
# Bin E: rem=90. Item 30. New rem=60. min(60,40)=40. Score -40.
# We prefer D (-30 > -40).
# D becomes 30 (half-full). E becomes 60 (more than half-full).
# Is this good?
# Polarization says: avoid the middle.
# If we pick D, we create a bin at 30.
# If we pick E, we create a bin at 60.
# 60 is further from the middle (50) than 30? No, 30 is dist 20 from 50. 60 is dist 10 from 50.
# Wait, the middle is 50.
# min(rem, cap-rem) is a measure of "how close to the middle".
# Max min(rem, cap-rem) is the middle.
# Min min(rem, cap-rem) is the edges.
#
# Let's re-evaluate D and E.
# D: rem 60 -> 30. min(30, 70) = 30.
# E: rem 90 -> 60. min(60, 40) = 40.
# We want to MINIMIZE min(rem, cap-rem).
# So we prefer D (30) over E (40).
# D is closer to the edge (0 or 100) than E?
# 30 is dist 30 from 0, 70 from 100. Min dist 30.
# 60 is dist 60 from 0, 40 from 100. Min dist 40.
# Yes, 30 is "more polarized" than 60.
#
# So the strategy is: Score = -min(rem_after, cap - rem_after).
# Higher score (less negative) is better.
# This pushes the resulting state towards the extremes.

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins based on polarization.
    bins: array of remaining capacities.
    Returns: array of scores. Higher is better.
    """
    # Filter bins that can fit the item
    valid_mask = bins >= item
    if not np.any(valid_mask):
        # No valid bins? The contract says "bins with enough remaining capacity".
        # If none, we probably return -inf for all, or let the caller handle it.
        # The prompt says: "scores each bin with enough remaining capacity".
        # It implies we only return scores for valid ones? 
        # "np.ndarray of one score per bin with enough remaining capacity"
        # But the signature says `-> np.ndarray` and the input is `bins`.
        # Usually in these contracts, we return an array of the same length, 
        # with -inf for invalid bins, so that `argmax` picks a valid one.
        # Let's assume we return same length array.
        scores = np.full_like(bins, -np.inf, dtype=float)
        return scores

    # Calculate remaining capacity after placing item
    rem_after = bins - item
    
    # Calculate "polarization" score: -min(rem, cap-rem)
    # We only care about valid bins. For invalid, score is -inf.
    # min(rem_after, CAPACITY - rem_after)
    # Since rem_after <= CAPACITY - item < CAPACITY, CAPACITY - rem_after > 0.
    # rem_after >= 0.
    
    # We can compute this for all bins, then mask.
    # But rem_after for invalid bins is negative.
    # min(negative, positive) = negative.
    # -min(negative, positive) = positive.
    # This would make invalid bins look good!
    # So we must mask.
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Only compute for valid bins to save time? Or just compute and mask.
    # Vectorized is faster.
    # rem_after for valid bins is >= 0.
    # We need min(rem_after, CAPACITY - rem_after).
    
    # Let's just compute for all, then set invalid to -inf.
    # For invalid bins, rem_after < 0.
    # min(rem_after, CAPACITY - rem_after) will be rem_after (negative).
    # Score = -rem_after (positive).
    # We will overwrite these with -inf.
    
    # Optimization: 
    # min(rem, cap-rem) is 0 at rem=0 and rem=cap.
    # It is max at rem=cap/2.
    # We want to minimize this value.
    
    # rem_after = bins - item
    # val = np.minimum(rem_after, CAPACITY - rem_after)
    # scores = -val
    
    # But for invalid bins, we want -inf.
    
    # Let's do it in one go.
    rem_after = bins - item
    # For valid bins, rem_after >= 0.
    # val = min(rem_after, CAPACITY - rem_after)
    # If rem_after is negative (invalid), val is negative.
    # We want scores to be -inf for invalid.
    
    # Calculate polarization score for all
    # Note: if rem_after > CAPACITY, that's impossible since bins <= CAPACITY.
    # So CAPACITY - rem_after is always >= item >= 0? 
    # No, if item is small, rem_after is large.
    # Example: item=1, bins=100. rem=99. min(99, 1) = 1.
    # Example: item=1, bins=50. rem=49. min(49, 51) = 49.
    
    val = np.minimum(rem_after, CAPACITY - rem_after)
    scores = -val
    
    # Mask invalid bins
    scores[~valid_mask] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
