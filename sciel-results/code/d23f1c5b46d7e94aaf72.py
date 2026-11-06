"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # The maximum non-coprime set containing 255 is all multiples of 3 in [1, 255]
    # Count: 255 // 3 = 85 elements
    # Any number not divisible by 3 cannot coexist with a multiple of 3 that isn't divisible by that number's prime factors
    # This is optimal: 85 elements
    
    result = list(range(3, 256, 3))
    
    # Verify: all elements are in [1, 255], 255 is included, and all pairs have gcd > 1
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    # All multiples of 3 share factor 3, so gcd >= 3 > 1 for all pairs
    
    return result
# EVOLVE-BLOCK-END
