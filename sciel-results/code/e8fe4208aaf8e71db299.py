# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin before 300s limit

def priority(vector, n):
    # Strongly prioritize vectors with fewer 1s (less likely to be midpoints)
    # Use fast prime-based hash for tie-breaking
    n1 = 0
    n2 = 0
    h = 0
    for x in vector:
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
        h = (h * 31 + x) & 0xFFFFFFFF
    
    # Primary: minimize count of 1s (strong weight)
    # Secondary: balance 0s and 2s
    n0 = n - n1 - n2
    
    # Strong penalty for 1s, mild penalty for imbalance of 0s/2s
    score = -3.0 * n1 - 0.1 * abs(n0 - n2)
    
    # Small hash-based tie-breaker
    hash_component = (h % 100000) / 100000.0 * 0.001
    
    return score + hash_component
# EVOLVE-BLOCK-END
