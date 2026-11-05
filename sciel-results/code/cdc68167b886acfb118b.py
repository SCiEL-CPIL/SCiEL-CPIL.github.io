import time
import math

def construct():
    """
    Constructs a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    For n = 255 = 3 * 5 * 17, the largest pairwise non-coprime set
    containing 255 is the set of all multiples of the smallest prime factor, 3.
    Any two distinct multiples of 3 have a gcd of at least 3.
    Size of this set is floor(255 / 3) = 85.
    """
    start_time = time.time()
    deadline = start_time + 60  # 60 seconds limit
    
    n = 255
    
    # Find prime factors of n
    # 255 = 3 * 5 * 17
    # Smallest prime factor is 3
    
    # The optimal strategy for square-free n is to take all multiples of the smallest prime factor.
    # This ensures gcd(a, b) >= p for all a, b in the set.
    # Any mixing of multiples of different prime factors introduces pairs with gcd 1.
    
    smallest_prime = 3
    result = list(range(smallest_prime, n + 1, smallest_prime))
    
    # Check deadline (though this is instant)
    if time.time() > deadline:
        # Fallback to a smaller valid set if we somehow timed out (impossible here)
        return [n]
        
    return result
