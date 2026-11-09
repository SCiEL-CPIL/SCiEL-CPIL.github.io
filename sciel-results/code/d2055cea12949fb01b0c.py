"""
Constructs a maximum-cardinality subset of [1, 255] containing 255
where every distinct pair has gcd > 1.

255 = 3 * 5 * 17. The set of all multiples of 3 in [1, 255] has size 85
and is a valid clique. Analysis suggests this is optimal because mixing
multiples of different primes (5, 17) forces the exclusion of many
multiples of 3 due to coprime conflicts, resulting in a smaller set.

This program returns the multiples of 3 directly.
"""

from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    # All multiples of 3 in [1, 255]
    # 255 is a multiple of 3 (255 = 3 * 85), so it is included.
    result = list(range(3, 256, 3))
    
    # The size is 85.
    # This is the known optimal solution for this specific instance.
    return result
