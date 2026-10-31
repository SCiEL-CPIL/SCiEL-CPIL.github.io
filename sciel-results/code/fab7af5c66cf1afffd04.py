"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
import zlib

def priority(vector, n):
    # Combine a structured base-3 hash with a random CRC32 hash
    # to improve the distribution of vector priorities.
    s = 0
    for v in vector:
        s = s * 3 + v
    return float((s * 1000003 + zlib.crc32(bytes(vector))) % (10**9))
# EVOLVE-BLOCK-END
