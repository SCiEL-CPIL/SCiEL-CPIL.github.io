import math
import time
from itertools import combinations

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0

    # 255 = 3 * 5 * 17
    # The largest pairwise non-coprime subset containing N is the set of multiples of the smallest prime factor of N.
    # Smallest prime factor of 255 is 3.
    # Set: {3, 6, ..., 255}
    # Size: 85
    
    # Verify that 255 is divisible by 3 (it is)
    # Verify that all pairs in this set have gcd > 1
    # gcd(3a, 3b) = 3*gcd(a,b) >= 3 > 1
    
    result = list(range(3, 256, 3))
    
    # Ensure 255 is in the set
    if 255 not in result:
        result.append(255)
        result.sort()
        
    return result
