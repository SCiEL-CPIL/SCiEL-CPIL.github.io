import math

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    
    # The maximum clique is the set of multiples of the smallest prime factor of N.
    # Prime factors of 255 are 3, 5, 17.
    # Multiples of 3: {3, 6, ..., 255}. Size 85.
    # Any other combination yields a smaller set.
    
    result = [x for x in range(1, N + 1) if x % 3 == 0]
    
    return result
