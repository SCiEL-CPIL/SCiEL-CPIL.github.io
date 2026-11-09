import time

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Combined Strategy: Balanced-Composition Routing with High-Quality Hashing.
    
    1. Structural Priority (Layering):
       We prioritize vectors based on how "balanced" their composition is.
       A vector is 'balanced' if the count of 1s is close to n/2.
       This routes the greedy search through the center of the affine cube,
       avoiding early saturation of corner subspaces (like {0,2}^n).
       Vectors with higher balance scores are always considered before
       vectors with lower balance scores.
       
    2. Intra-Layer Ordering (Refinement):
       Within each balance layer (vectors with the same count of 1s),
       we use a high-quality deterministic pseudo-random ordering derived
       from a MurmurHash3-style mixing function. This ensures that the
       specific order in which points from the same layer are considered
       is uncorrelated and avoids systematic biases that could lead to
       suboptimal local configurations.
       
    The final priority is a float: layer_score + hash_fraction.
    Since layer_score is an integer and hash_fraction is in [0, 1),
    this guarantees strict ordering by layer, with hash breaking ties.
    """
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Calculate structural balance score
    # Count the number of 1s in the vector
    count_1 = 0
    for v in vector:
        if v == 1:
            count_1 += 1
    
    # Target is n/2. 
    # Score is high when count_1 is close to n/2.
    # Formula: n - abs(2 * count_1 - n)
    # If n=8, count_1=4 -> 8 - 0 = 8 (Max)
    # If n=8, count_1=0 -> 8 - 8 = 0 (Min)
    layer_score = n - abs(2 * count_1 - n)
    
    # 2. Calculate high-quality hash for tie-breaking
    # Convert vector to a single integer index
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    # Mix idx and n using MurmurHash3 finalizer logic
    # This ensures that different 'n' values produce uncorrelated orders
    # and that the order within a layer is pseudo-random.
    
    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    # Combine idx and n to decorrelate orders for different dimensions
    # Using XOR with a scaled golden ratio constant
    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    # Normalize hash to [0, 1)
    # 2**64 is the range of the 64-bit integer
    hash_fraction = mixed / (2**64)
    
    # 3. Combine
    # layer_score is an integer. hash_fraction is in [0, 1).
    # The sum ensures that any vector in layer L has priority in [L, L+1),
    # and any vector in layer L+1 has priority in [L+1, L+2).
    # Thus, all vectors in higher layers are strictly preferred.
    # Within a layer, the order is determined by the hash.
    
    return float(layer_score + hash_fraction)
