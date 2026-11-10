import math

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    
    The optimal set is the set of all multiples of 3 in [1, 255].
    Size is floor(255/3) = 85.
    """
    N = 255
    # The set of all multiples of the smallest prime factor of N (3)
    # is pairwise non-coprime and contains N.
    # This is the maximum possible size.
    return list(range(3, N + 1, 3))
