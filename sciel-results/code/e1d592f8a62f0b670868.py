"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
from math import gcd
from functools import reduce

def _pairwise_coprime(nums):
    """Check if all distinct values in nums are pairwise coprime."""
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    # We want to find A, B such that all 9 sums a+b are distinct primes.
    # This guarantees pairwise coprimality and max cardinality 9.
    
    # Precompute primes up to 256
    primes = set()
    for i in range(2, 257):
        if _is_prime(i):
            primes.add(i)
            
    # Strategy:
    # Fix a_1 = 1.
    # Then b_j must be such that 1 + b_j is prime.
    # Let V_b be the set of b in [1, 128] such that 1+b is prime.
    
    V_b = [b for b in range(1, 129) if (1 + b) in primes]
    
    # Iterate over combinations of 3 from V_b for B
    for B in itertools.combinations(V_b, 3):
        B = list(B)
        # For this B, find valid a's such that a+b is prime for all b in B
        # a must be in [1, 128] and distinct from 1 (since a_1=1)
        # Actually, A must have 3 distinct elements. We fixed a_1=1.
        # So we need a_2, a_3 from [2, 128] such that a_2+b in primes and a_3+b in primes for all b in B.
        
        valid_as = []
        for a in range(2, 129):
            if all((a + b) in primes for b in B):
                valid_as.append(a)
                
        # We need to pick 2 distinct elements from valid_as
        if len(valid_as) >= 2:
            for A_rest in itertools.combinations(valid_as, 2):
                A = [1] + list(A_rest)
                # Check if all 9 sums are distinct
                sums = [a + b for a in A for b in B]
                if len(set(sums)) == 9:
                    return [A, B]
                    
    # If the prime strategy fails, fall back to the previous search or a broader search
    # But prime strategy should work.
    
    # Fallback: Random search for pairwise coprime sums (not necessarily prime)
    import random
    random.seed(42)
    range_vals = list(range(1, 129))
    best_A = None
    best_B = None
    best_count = 0
    
    for _ in range(20000):
        A = sorted(random.sample(range_vals, 3))
        B = sorted(random.sample(range_vals, 3))
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        if count > best_count:
            if _pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]
                    
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
