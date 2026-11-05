import math
import time

def construct():
    n = 255
    # The set of all multiples of 3 in [1, 255] is a valid set.
    # 255 is a multiple of 3.
    # Any two multiples of 3 have gcd at least 3.
    # Size is 255 // 3 = 85.
    # Analysis suggests this is the maximum possible size.
    # Returning this set directly is O(N) and avoids search timeouts.
    
    result = list(range(3, n + 1, 3))
    
    # Verify 255 is in the set
    if n not in result:
        result.append(n)
        result.sort()
        
    return result
