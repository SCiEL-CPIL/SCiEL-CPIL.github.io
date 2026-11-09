"""Optimized OneMax constructor; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # Construct an integer with the lower N bits set to 1
    # (2^N - 1) has N ones in binary representation
    val = (1 << N) - 1
    # Format as binary string with zero padding to ensure length N
    return format(val, '0{}b'.format(N))
# EVOLVE-BLOCK-END
