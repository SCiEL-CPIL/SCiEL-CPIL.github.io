import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def _mix64(h):
    """64-bit MurmurHash3 finalizer for high-quality mixing."""
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

# Precompute weights for the "spectral" penalty.
# The idea is to penalize vectors that are "low frequency" in a certain sense.
# In F_3^n, the all-ones vector (1,1,...,1) and its multiples are low frequency.
# Vectors with many 1s are "closer" to the center.
# Let's define a penalty based on the deviation from a "random" distribution of 1s.
# Or better, let's look at the "energy" of the vector.
# A vector v has "energy" sum(v_i^2).
# For {0,1,2}, 0^2=0, 1^2=1, 2^2=4.
# If we want to penalize low frequency (correlation with all-ones), we might look at sum(v_i).
# But the inspiration suggests penalizing high correlation to the all-ones vector.
# The all-ones vector is (1,1,...,1).
# Correlation is proportional to sum(v_i - 1).
# Let's try a different structural feature:
# The number of 1s is a good proxy for "distance from corners".
# But let's combine it with a hash-based perturbation that is scaled by the "rarity" of the composition.
# This ensures that rare compositions (like all 1s or all 0s) get more hash-driven randomness,
# while common compositions (like balanced ones) are more strictly ordered by their structural score.

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Structural Score: Minimize 2s (Prioritize {0,1}^n subspace)
    # Hypothesis: {0,1}^n is a cap set. By prioritizing vectors without 2s,
    # we ensure the entire {0,1}^n subspace is filled first.
    # This guarantees a cap set size of at least 2^n.
    # Vectors with 2s are more likely to form 3-APs with {0,1}^n vectors,
    # so delaying them avoids "blocking" the safe subspace.
    
    count_2 = 0
    for v in vector:
        if v == 2:
            count_2 += 1
            
    # Strong penalty for 2s.
    # This creates layers: Layer 0 (no 2s), Layer 1 (one 2), etc.
    structural_score = -1000.0 * count_2
    
    # Secondary: Within the same count_2 layer, we can use a hash or another feature.
    # For now, let's add a small penalty for 1s as well?
    # No, {0,1}^n is safe.
    # What about {0,2}^n? Also safe.
    # But we can't prioritize both.
    # {0,1}^n is as good as {0,2}^n.
    # Let's stick to minimizing 2s.

    # 2. Hash-based Perturbation
    # Compute the index in base 3.
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v
    
    # Mix with n to ensure different n's have different orderings.
    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = _mix64(seed_val)
    
    # Scale hash to [0, 1)
    hash_fraction = mixed / (2**64)
    
    # Combine structural score and hash.
    # We want the structural score to dominate.
    # But we want the hash to break ties effectively.
    # Also, we want to introduce some "noise" into the structural score to avoid rigid layers.
    # Let's scale the hash by a factor that depends on the "rarity" or "balance".
    # If the composition is very common, the layer is large, so we need good mixing.
    # If the composition is rare, the layer is small, so mixing is less critical.
    
    # Let's use a fixed small perturbation.
    # perturbation = hash_fraction * 0.1
    # result = structural_score + perturbation
    
    # Let's try a different combination:
    # result = structural_score * 10 + hash_fraction
    # This ensures that a difference of 1 in structural score is always larger than any hash difference.
    
    result = structural_score * 10.0 + hash_fraction
    
    return float(result)

# EVOLVE-BLOCK-END
