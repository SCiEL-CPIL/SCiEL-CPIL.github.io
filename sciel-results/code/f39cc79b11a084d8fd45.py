import time

_START_TIME = time.time()
_DEADLINE = 330.0  # Seconds

def priority(vector, n):
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback to a very fast simple score
        return -sum(v * v for v in vector)
    
    # Bitwise parity-based priority:
    # 1. Convert vector to base-3 integer
    val = 0
    for v in vector:
        val = val * 3 + v
    
    # 2. Generate a mask based on n to ensure variety across different dimensions
    # Using a simple deterministic hash of n
    mask_seed = n * 2654435761 % (2**32)
    # XOR with a fixed pattern to ensure good bit distribution
    mask = 0x5A5A5A5A ^ mask_seed
    
    # 3. XOR the base-3 integer with the mask
    xor_val = val ^ mask
    
    # 4. Count bits (popcount)
    try:
        bits = xor_val.bit_count()
    except AttributeError:
        bits = bin(xor_val).count('1')
    
    # 5. Add a secondary term based on sum of squares to break ties
    # and provide a structural preference for higher-energy vectors
    sum_sq = 0
    for v in vector:
        sum_sq += v * v
    
    # 6. Add a small polynomial hash for deterministic tie-breaking
    h = 0
    base = 31
    for v in vector:
        h = (h * base + v) % (10**9 + 7)
    h_norm = h / (10**9 + 7)
    
    # Combine: primary is bit count (chaotic interleaving),
    # secondary is sum of squares (structural),
    # tertiary is hash (tie-breaking)
    return float(bits) + 0.01 * sum_sq + h_norm
