import time
import math
from collections import Counter

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

# Precomputed reference vectors for soft-min distance computation.
# These are derived from a bitwise parity scheme on ternary representations.
# We use a set of "anchor" vectors that tile the space chaotically.
# For each coordinate position, we use a fixed mask to create diversity.
_ANCHOR_COUNT = 16
_anchors = []

def _build_anchors(n):
    """Build anchor vectors using bitwise parity on ternary integers."""
    global _anchors
    if _anchors:
        return
    # Create anchors by interpreting a counter as a ternary number
    # and XORing with various masks to get chaotic but deterministic vectors
    for mask_id in range(_ANCHOR_COUNT):
        vec = []
        for i in range(n):
            # Bitwise parity: use bit (mask_id + i) of a golden-ratio-derived constant
            # This creates a chaotic interleaving that breaks local correlations
            val = ((mask_id * 7 + i * 13 + (i * i) % 5) * 31) % 3
            vec.append(val)
        _anchors.append(tuple(vec))

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Soft-min Hamming distance to a set of chaotic anchor vectors
       (log-sum-exp surrogate for the nonsmooth min-distance objective).
       This spreads selections without O(k) tracking of all selected vectors.
    2. Secondary: Prioritize vectors with all 3 distinct symbols.
    3. Tertiary: Balance - symbol counts close to uniform.
    4. Quaternary: Bitwise parity hash for deterministic tie-breaking.
    """
    global _start_time, _anchors
    
    if _start_time is None:
        _start_time = time.time()
        _build_anchors(n)
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    vec = vector  # use as-is, assume list or tuple of ints
    
    # Fast symbol counting
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vec:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Distinct count score
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0 or c0 > 0 and c2 > 0 or c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Balance: L1 deviation from uniform
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Soft-min distance to anchors using log-sum-exp
    # This is a smooth surrogate for min_hamming_distance(vec, anchor)
    # We want to MAXIMIZE the soft-min distance (push away from anchors)
    # but actually we want to AVOID being close to already-selected regions.
    # Since we don't track selected vectors, we use anchors as a proxy for
    # "already occupied" regions. Vectors close to anchors get penalized.
    # Actually, for cap sets, we want vectors that are FAR from each other.
    # The anchors represent a fixed grid; we want vectors that are NOT
    # aligned with this grid structure.
    
    # Compute Hamming distances to each anchor
    dists = []
    for anchor in _anchors:
        d = 0
        for a, b in zip(vec, anchor):
            if a != b:
                d += 1
        dists.append(d)
    
    # Soft-min via log-sum-exp: soft_min(d) = -log(sum(exp(-d_i / T))) * T
    # Lower soft-min means closer to some anchor (worse for spread)
    # We want to MAXIMIZE distance, so we negate
    T = 2.0  # temperature for smoothness
    neg_dists = [-d / T for d in dists]
    max_nd = max(neg_dists)
    sum_exp = sum(math.exp(nd - max_nd) for nd in neg_dists)
    soft_min_val = -T * (max_nd + math.log(sum_exp))
    # soft_min_val is the soft minimum distance (positive)
    # We want to maximize this, so:
    distance_score = soft_min_val * 50.0
    
    # Bitwise parity hash for tie-breaking
    # Treat vector as ternary integer, XOR with mask, count "bits"
    h = 0
    for i, val in enumerate(vec):
        h = (h * 31 + val + i * 7) & 0x7FFFFFFF
    # Bit parity of the hash
    bit_count = bin(h).count('1')
    hash_score = bit_count / 31.0  # normalize to ~[0,1]
    
    # Combine
    return float(distance_score + distinct_score + balance_score + hash_score)
