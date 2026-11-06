"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def priority(vector, n):
    """
    Assign a deterministic, fast pseudo-random priority using a custom 64-bit hash.
    This avoids the overhead of cryptographic hashing while maintaining
    a high-quality, uniform distribution over the hypercube.
    """
    # Initial seed
    h = 0x87C37B9111E2A4F9
    for x in vector:
        # Mix in the element
        h ^= x
        # Multiply by a large odd constant
        h = (h * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    # Final avalanche (Mix the bits thoroughly)
    h ^= (h >> 33)
    h = (h * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    
    # Map to [0, 1)
    return h / (1 << 64)
# EVOLVE-BLOCK-END
