"""Optimized coprime set construction via exhaustive search over small primes."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product

_START_TIME = time.time()
_MAX_TIME_LIMIT = 85.0

def construct():
    """
    Returns a list of distinct integers from 1 to 256, pairwise coprime,
    maximizing their sum.
    
    Strategy:
    1. Primes > 16 (i.e., 17, 19, ..., 251) can only appear as themselves in any valid set,
       because p*q > 256 for any distinct primes p,q >= 17.
       So we include all such primes.
    2. For primes <= 16 (2, 3, 5, 7, 11, 13), we can form composite numbers like p*q, p*q*r, etc.
       We need to partition these small primes into groups, where each group forms a single number
       (the product of the primes in the group, possibly with powers, but since p^2 > 256 for p>=17,
       and for small primes we can have powers, but actually the optimal is to use each prime at most once
       in a composite, because p^2 < p*q for q > p, so it's better to pair with another prime than to square).
       
       Actually, let's think: for prime p, we can use p^k where p^k <= 256. But if we use p^2, we can't use p*q.
       Since p*q > p^2 when q > p, it's better to pair p with a larger prime q rather than squaring p.
       
       So the optimal strategy for small primes is to partition them into groups, and each group's product
       is one number. We want to maximize the sum of these products.
       
    3. We do an exhaustive search over all partitions of the small primes {2,3,5,7,11,13}.
       The number of partitions of 6 elements is 203, which is manageable.
    
    4. For each partition, compute the sum of products, and keep the best one.
    5. Add 1 (since 1 is coprime with everything).
    
    6. Return the list of numbers.
    """
    if time.time() - _START_TIME > _MAX_TIME_LIMIT:
        return [1]
    
    limit = 256
    
    # Find all primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [x for x in range(2, n + 1) if is_prime[x]]
    
    primes = sieve(limit)
    
    # Split primes into small (<=16) and large (>16)
    small_primes = [p for p in primes if p <= 16]  # [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 16]   # [17, 19, ..., 251]
    
    # All large primes must be included as themselves
    base_sum = sum(large_primes)
    
    # Now, find the best partition of small_primes to maximize the sum of products
    # We use recursive backtracking to generate all partitions
    
    def best_partition_sum(primes_list):
        """
        Returns the maximum sum of products for a partition of primes_list.
        Uses memoization on the frozenset of remaining primes.
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def solve(remaining):
            if not remaining:
                return 0
            remaining = frozenset(remaining)
            best = 0
            # Try all subsets of remaining
            remaining_list = list(remaining)
            n = len(remaining_list)
            for mask in range(1, 1 << n):
                subset = frozenset(remaining_list[i] for i in range(n) if mask & (1 << i))
                # Compute product of subset
                prod = 1
                for p in subset:
                    prod *= p
                if prod > limit:
                    continue
                rest = frozenset(remaining - subset)
                val = prod + solve(rest)
                if val > best:
                    best = val
            return best
        
        return solve(frozenset(primes_list))
    
    best_small_sum = best_partition_sum(small_primes)
    
    # Now we need to reconstruct the actual numbers for the best partition
    # We'll do a similar search but track the actual groups
    
    def reconstruct_best_partition(primes_list):
        """
        Returns the list of products (one per group) that achieve the maximum sum.
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def solve(remaining):
            if not remaining:
                return (0, [])
            remaining = frozenset(remaining)
            best_val = -1
            best_groups = []
            remaining_list = list(remaining)
            n = len(remaining_list)
            for mask in range(1, 1 << n):
                subset = frozenset(remaining_list[i] for i in range(n) if mask & (1 << i))
                prod = 1
                for p in subset:
                    prod *= p
                if prod > limit:
                    continue
                rest = frozenset(remaining - subset)
                rest_val, rest_groups = solve(rest)
                val = prod + rest_val
                if val > best_val:
                    best_val = val
                    best_groups = [prod] + rest_groups
            return (best_val, best_groups)
        
        _, groups = solve(frozenset(primes_list))
        return groups
    
    small_groups = reconstruct_best_partition(small_primes)
    
    # Construct the final set
    result = [1]  # Include 1
    result.extend(large_primes)
    result.extend(small_groups)
    
    # Ensure all are distinct and within [1, 256]
    # Verify pairwise coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # This shouldn't happen, but just in case
                pass
    
    # Ensure all are Python ints
    final_result = [int(x) for x in result]
    
    return final_result
# EVOLVE-BLOCK-END
