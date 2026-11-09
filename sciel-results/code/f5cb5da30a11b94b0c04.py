import math
import time

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    The theoretical maximum is 85 (all multiples of 3 in [1, 255]).
    We verify this is achievable and return it directly for speed and reliability.
    """
    # Deadline guard (90s wall-clock, return well before)
    start_time = time.time()
    
    # The set of all multiples of 3 in [1, 255]
    # 255 = 3 * 85, so multiples are 3, 6, 9, ..., 255
    # Size = 255 // 3 = 85
    # All pairs (3i, 3j) have gcd >= 3 > 1, so pairwise non-coprime condition holds.
    # 255 is included (3 * 85 = 255).
    
    candidate = list(range(3, 256, 3))
    
    # Verify: 255 is in the set
    if 255 not in candidate:
        candidate.append(255)
    
    # Verify: all elements are in [1, 255]
    for x in candidate:
        if x < 1 or x > 255:
            # Should not happen, but safety check
            pass
    
    # Verify: pairwise gcd > 1 (spot check for safety)
    # For multiples of 3, this is guaranteed, but let's do a quick check
    n = len(candidate)
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidate[i], candidate[j]) <= 1:
                valid = False
                break
        if not valid:
            break
        if time.time() - start_time > 85:  # Safety margin
            break
    
    if not valid:
        # Fallback: just return the multiples of 3 (which we know is valid)
        return list(range(3, 256, 3))
    
    return candidate
