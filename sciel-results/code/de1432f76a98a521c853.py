"""Program for the suite task: maximize a non-coprime subset of [1,255] containing 255."""

def construct():
    """
    Return the maximum cardinality subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    The optimal strategy is to take all multiples of the smallest prime factor of 255.
    255 = 3 * 5 * 17, so the smallest prime factor is 3.
    All multiples of 3 in [1, 255] form a valid set because any two distinct multiples
    of 3 have gcd at least 3.
    
    Size: floor(255 / 3) = 85.
    """
    return list(range(3, 256, 3))
