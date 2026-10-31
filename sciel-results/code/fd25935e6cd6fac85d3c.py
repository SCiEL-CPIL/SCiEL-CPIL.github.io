"""Cap set greedy construction with randomized priority via SHA256."""
import hashlib
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Convert vector to a canonical string representation
    key = ''.join(str(x) for x in vector)
    # Use SHA256 to get a 64-bit hash for uniform random ordering
    h = hashlib.sha256(key.encode('utf-8')).digest()
    # Take the first 8 bytes as a 64-bit integer
    val = int.from_bytes(h[:8], byteorder='big')
    # Normalize to [0, 1)
    return val / (2**64)
# EVOLVE-BLOCK-END
