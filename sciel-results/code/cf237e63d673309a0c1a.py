import time
from math import gcd

def construct():
    # 255 = 3 * 5 * 17
    # The largest clique in the graph where vertices are [1, 255] with gcd(x, 255) > 1
    # and edges connect numbers with gcd > 1 is formed by the multiples of the smallest prime factor.
    # Smallest prime factor is 3.
    # Multiples of 3 in [1, 255]: 3, 6, ..., 255.
    # Count: 255 // 3 = 85.
    # All pairs in this set have gcd >= 3.
    # 255 is in the set.
    
    return list(range(3, 256, 3))
