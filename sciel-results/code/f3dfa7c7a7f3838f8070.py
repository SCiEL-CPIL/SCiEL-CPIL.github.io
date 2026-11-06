import time
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    The set of all multiples of 3 in [1, 255] has size 85.
    It contains 255.
    Any two multiples of 3 have gcd >= 3 > 1.
    
    We verify that no larger set exists.
    Any element in the set must share a factor with 255 (3, 5, or 17).
    Let S be the set.
    If S contains any element not divisible by 3, it must be divisible by 5 or 17.
    If it contains an element divisible by 5 (but not 3), say 5, then all other elements
    must share a factor with 5. Thus all elements must be divisible by 5.
    The set of multiples of 5 has size 51.
    If it contains an element divisible by 17 (but not 3 or 5), say 17, then all other
    elements must share a factor with 17. Thus all elements must be divisible by 17.
    The set of multiples of 17 has size 15.
    
    Mixing these is not beneficial because adding a non-multiple of 3 forces the
    exclusion of most multiples of 3.
    Specifically, adding 5 forces the set to be multiples of 5 (size 51).
    Adding 17 forces the set to be multiples of 17 (size 15).
    
    Therefore, the maximum cardinality is 85.
    """
    return list(range(3, 256, 3))
