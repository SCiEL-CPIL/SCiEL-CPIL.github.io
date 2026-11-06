"""Optimized construction for large pairwise noncoprime sets containing 255."""
import math
import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # The set must contain 255.
    # For any x in the set, gcd(x, 255) > 1, so x must be divisible by 3, 5, or 17.
    # For any pair a, b in the set, gcd(a, b) > 1.
    
    # If the set contains 3, then for any other x, gcd(3, x) > 1 implies 3 divides x.
    # Thus, if 3 is in the set, all elements must be multiples of 3.
    # The number of multiples of 3 in [1, 255] is floor(255/3) = 85.
    
    # If the set contains 5 but not 3, then for any other x, gcd(5, x) > 1 implies 5 divides x.
    # Thus, if 5 is in the set (and 3 is not, otherwise we are in the previous case), 
    # all elements must be multiples of 5.
    # The number of multiples of 5 in [1, 255] is floor(255/5) = 51.
    
    # If the set contains 17 but not 3 or 5, all elements must be multiples of 17.
    # The number of multiples of 17 in [1, 255] is floor(255/17) = 15.
    
    # What if the set contains none of {3, 5, 17}?
    # Then it must contain other multiples.
    # If it contains 9, then for any x, gcd(9, x) > 1 implies 3 divides x.
    # So if 9 is in the set, all elements are multiples of 3.
    # We can have at most 85 elements (if we include 3) or 84 (if we exclude 3 but include 9).
    # Since 85 > 84, including 3 is better if we are restricted to multiples of 3.
    
    # Generally, the constraint "gcd(a, b) > 1 for all pairs" forces the set to be a subset
    # of the multiples of some prime p that divides at least one element in a "core" way.
    # Specifically, if p is the smallest prime factor of any element in the set, 
    # and that element is p itself, then all elements must be multiples of p.
    # If the element is p^k, it still forces divisibility by p for compatibility with other primes?
    # Not necessarily all, but it severely restricts the set.
    # However, the set of ALL multiples of the most frequent prime factor of 255 is the largest valid clique.
    
    # Primes dividing 255: 3, 5, 17.
    # Counts:
    # Multiples of 3: 85
    # Multiples of 5: 51
    # Multiples of 17: 15
    
    # The maximum size is 85.
    
    # Construct the set of all multiples of 3 in [1, 255]
    result = list(range(3, 256, 3))
    
    # Verification (optional but good for safety)
    if time.time() < deadline:
        assert 255 in result
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if math.gcd(result[i], result[j]) <= 1:
                    # This should never happen for multiples of 3
                    break
    
    return result

# EVOLVE-BLOCK-END
