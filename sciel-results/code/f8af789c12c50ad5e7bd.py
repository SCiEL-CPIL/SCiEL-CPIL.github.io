import time
import zlib
import math

_START_TIME = time.time()
TIME_LIMIT = 280

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Structural change:
    1. Maintains global digit balance (primary).
    2. Adds bias towards zeros (additive identity safety).
    3. Penalizes spatial clustering of non-zero digits (spread).
    4. Uses a fast CRC32 hash for deterministic tie-breaking.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Primary: Minimize deviation from balanced distribution
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Secondary: Bias towards zeros (Directive)
    # Vectors with more zeros are structurally safer
    zero_bias = c0 * 10.0
    
    # Tertiary: Penalize spatial clustering of non-zeros
    # Clustering non-zeros might create local density that favors APs
    clustering = 0
    for i in range(n - 1):
        if vector[i] != 0 and vector[i + 1] != 0:
            clustering += 1
    spread_penalty = clustering * 5.0
    
    # Quaternary: Fast hash tie-breaker
    # Convert to bytes for CRC32
    vec_bytes = bytes(vector)
    hash_val = zlib.crc32(vec_bytes) / (2**32)
    
    # Combine
    # Balance is most important, then zero bias, then spread, then hash
    return -penalty * 10000.0 + zero_bias - spread_penalty + hash_val * 100.0
