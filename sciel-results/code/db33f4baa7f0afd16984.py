"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assign a deterministic, fast pseudo-random priority to each vector in {0,1,2}^n.
    Uses a fast integer mixing function (SplitMix64) on the base-3 representation
    of the vector. This is significantly faster than SHA-256, allowing for
    faster evaluation and potentially exploring larger n or finishing within
    time limits more reliably. The high-quality mix ensures a uniform
    pseudo-random permutation of the space.
    """
    # Convert vector to a single integer in base 3
    x = 0
    for v in vector:
        x = x * 3 + v
    
    # Apply SplitMix64 integer mixing function
    # This ensures good distribution of the hash values
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb
    x = (x & 0xFFFFFFFFFFFFFFFF)
    x = x ^ (x >> 31)
    x = x & 0xFFFFFFFFFFFFFFFF
    
    # Map to [0, 1)
    return x / (1 << 64)
# EVOLVE-BLOCK-END
