# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic polynomial hash to create a pseudo-random ordering.
    # This avoids the bad clustering of lexicographic order.
    # Based on knowledge that polynomial hash outperformed CRC32.
    
    # Horner's method: h = (h * base + v_i) % MOD
    base = 31
    mod = (1 << 61) - 1 # Large prime
    
    h = 0
    for v in vector:
        h = (h * base + v) % mod
        
    # Normalize to [0, 1)
    return h / mod
# EVOLVE-BLOCK-END
