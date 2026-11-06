import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85  # seconds, leave 5s margin

    # The problem asks for the largest subset of [1, 255] containing 255
    # such that gcd(a, b) > 1 for all distinct pairs.
    
    # 255 = 3 * 5 * 17.
    # Any element x in the set must have gcd(x, 255) > 1.
    # Thus x must be divisible by 3, 5, or 17.
    
    # A sufficient condition for a set to be pairwise non-coprime is that
    # all elements share a common prime factor.
    # The set of all multiples of the smallest prime factor of 255 (which is 3)
    # forms a valid clique.
    
    # Size of multiples of 3 in [1, 255]: floor(255 / 3) = 85.
    # Size of multiples of 5 in [1, 255]: floor(255 / 5) = 51.
    # Size of multiples of 17 in [1, 255]: floor(255 / 17) = 15.
    
    # It can be proven that the maximum clique size is determined by the
    # smallest prime factor. If we try to include a number not divisible by 3,
    # it cannot be coprime to 3 (which is in the set of multiples of 3).
    # Thus, we cannot mix multiples of 5 (not div by 3) with the full set of
    # multiples of 3.
    
    # Constructing the set of multiples of 3:
    S = list(range(3, 256, 3))
    
    # 255 is 3 * 85, so it is the last element in this range.
    # Ensure 255 is present (it is by construction).
    if 255 not in S:
        S.append(255)
        S = sorted(set(S))
        
    # Quick validity check (optional but good for robustness)
    # Since all elements are multiples of 3, gcd(a,b) >= 3 for all pairs.
    # This is O(N^2) but N=85, so 7225 checks, very fast.
    # We can skip full check for speed, but let's do a spot check or trust the math.
    # The math is sound: all elements are divisible by 3.
    
    # Return the list
    return S
