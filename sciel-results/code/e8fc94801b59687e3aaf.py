"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _is_pairwise_coprime(nums):
    """Check if all distinct values in nums are pairwise coprime."""
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    max_val = 128
    max_sum = 2 * max_val
    
    # Sieve of Eratosthenes for primes up to max_sum
    sieve = [True] * (max_sum + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_sum + 1, i):
                sieve[j] = False
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Strategy: Search for 3x3 grids where all 9 sums are distinct primes.
    # This guarantees pairwise coprimality and maximum cardinality (9).
    # 
    # Precompute for each x in [1,128] the list of y in [1,128] 
    # such that x+y is prime.
    prime_partner = [[] for _ in range(max_val + 1)]
    for x in range(1, max_val + 1):
        for y in range(1, max_val + 1):
            s = x + y
            if s <= max_sum and sieve[s]:
                prime_partner[x].append(y)
    
    # Iterate over a1, b1 such that a1+b1 is prime.
    # For each, try to find a2,a3,b2,b3 such that all 9 sums are prime.
    
    for a1 in range(1, max_val + 1):
        for b1 in range(1, max_val + 1):
            if a1 + b1 > max_sum or not sieve[a1 + b1]:
                continue
            
            # Candidates for b2,b3: values y != b1 such that a1+y is prime
            b_cands = [y for y in prime_partner[a1] if y != b1]
            
            # Candidates for a2,a3: values x != a1 such that x+b1 is prime
            a_cands = [x for x in prime_partner[b1] if x != a1]
            
            # Try all pairs (b2,b3) and (a2,a3)
            for b2 in b_cands:
                for b3 in b_cands:
                    if b2 == b3:
                        continue
                    for a2 in a_cands:
                        for a3 in a_cands:
                            if a2 == a3:
                                continue
                            
                            # Check remaining 4 sums are prime
                            if (sieve[a2 + b2] and sieve[a2 + b3] and 
                                sieve[a3 + b2] and sieve[a3 + b3]):
                                
                                A = [a1, a2, a3]
                                B = [b1, b2, b3]
                                sums = set()
                                for x in A:
                                    for y in B:
                                        sums.add(x + y)
                                
                                if len(sums) == 9:
                                    return [A, B]
                                elif len(sums) > best_count:
                                    if _is_pairwise_coprime(list(sums)):
                                        best_count = len(sums)
                                        best_A = A[:]
                                        best_B = B[:]
        
        if time.time() > deadline:
            if best_A is not None:
                return [best_A, best_B]
            return [[1, 7, 13], [4, 10, 16]]
    
    if best_A is not None:
        return [best_A, best_B]
    
    # Fallback: random search
    import random
    random.seed(42)
    for _ in range(200000):
        if time.time() > deadline:
            break
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        sums = set()
        for x in A:
            for y in B:
                sums.add(x + y)
        if len(sums) > best_count:
            if _is_pairwise_coprime(list(sums)):
                best_count = len(sums)
                best_A = A[:]
                best_B = B[:]
    
    if best_A is not None:
        return [best_A, best_B]
    
    return [[1, 7, 13], [4, 10, 16]]

# EVOLVE-BLOCK-END
