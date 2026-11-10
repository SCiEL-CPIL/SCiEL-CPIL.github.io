import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    # The optimal set for N=255 containing N is the set of all multiples
    # of the smallest prime factor of N, which is 3.
    # Size is floor(255/3) = 85.
    # Any other construction (like multiples of 5 or 17) is smaller.
    # Mixing branches requires removing elements, reducing size.
    return list(range(3, 256, 3))
