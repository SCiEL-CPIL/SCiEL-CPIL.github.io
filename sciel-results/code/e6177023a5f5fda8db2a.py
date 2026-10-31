"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a combined 64-bit hash to create a more uniform random ordering.
    # Combining CRC32 of the vector and its reverse increases entropy
    # and reduces the chance of correlated bad ordering patterns.
    b = bytes(vector)
    h1 = zlib.crc32(b)
    h2 = zlib.crc32(b[::-1])
    return float((h1 << 32) | h2)
# EVOLVE-BLOCK-END
