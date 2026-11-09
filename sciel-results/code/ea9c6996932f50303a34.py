# EVOLVE-BLOCK-START
import time
import sys

# Strategy: Combinatorial Hash-Scrambling with Weak Structural Bias
# 
# Rationale:
# The previous approach used a smooth heuristic (count of 1s) which creates 
# large contiguous blocks of similar vectors. When the greedy algorithm processes 
# these blocks, it often fills a subspace (like {0,2}^k) too densely, creating 
# many "midpoints" that block subsequent vectors with 1s.
#
# By using a high-entropy hash as the primary sort key, we interleave vectors 
# from different geometric regions of the space. This "scrambling" prevents 
# the greedy algorithm from systematically over-filling any local region before 
# considering vectors from other regions. It mimics the benefits of randomized 
# restarts but in a single, deterministic pass.
#
# The structural bias is kept very weak (small weight) so that the hash 
# dominates the ordering, ensuring the "scramble" effect is strong, while 
# still providing a slight preference for "balanced" vectors (which are 
# generally safer to add early than extreme vectors like all-0s or all-2s).

_START_TIME = time.time()
_TIMEOUT = 300.0  # Safety margin for the 330s limit

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def priority(vector, n):
    # Check time limit - if we are close to the deadline, return a neutral score
    # to avoid overhead. In practice, priority is called O(3^n) times, 
    # and this check is O(1).
    if _time_left() < 1.0:
        return 0.0

    # 1. Compute a high-entropy hash of the vector.
    # We use a polynomial rolling hash with a large prime modulus.
    # This ensures that vectors with similar structure (e.g., adjacent in Hamming distance)
    # are likely to have very different hash values, breaking local correlations.
    h = 0
    # Use a large prime for better distribution
    MOD = 2147483647  # Mersenne prime
    BASE = 31
    
    for i, v in enumerate(vector):
        h = (h * BASE + v) % MOD
        
    # Normalize hash to [0, 1)
    hash_score = h / MOD

    # 2. Compute a weak structural bias.
    # We want to slightly favor vectors that are "balanced" in terms of digit counts.
    # Specifically, vectors with counts of 0, 1, 2 all close to n/3 are often 
    # "central" in the cap set structure.
    # However, we keep this weight very small (e.g., 0.01) so the hash dominates.
    
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0: c0 += 1
        elif v == 1: c1 += 1
        else: c2 += 1
        
    # Ideal count is n/3
    target = n / 3.0
    # Penalty for deviation from target
    # This is a small quadratic penalty
    deviation = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    # Normalize deviation to roughly [0, 1] range for small n
    # Max deviation is roughly 3 * (n/2)^2 = 3n^2/4
    # Let's just use a small constant factor
    structural_score = -0.001 * deviation / (n * n + 1)

    # 3. Combine.
    # Hash score is in [0, 1).
    # Structural score is a small negative number.
    # We want higher priority for higher hash score (arbitrary but consistent) 
    # and less negative structural score.
    # Actually, let's make the hash the primary driver.
    # Since the hash is uniform, the order is effectively random but deterministic.
    
    final_score = hash_score + structural_score

    return final_score

# EVOLVE-BLOCK-END
