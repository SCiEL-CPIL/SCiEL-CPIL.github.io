"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START

# Precompute a list of primes up to 256 (max sum 128+128)
MAX_SUM = 256
_primes = []
_is_prime = [True] * (MAX_SUM + 1)
_is_prime[0] = _is_prime[1] = False
for _i in range(2, int(MAX_SUM**0.5) + 1):
    if _is_prime[_i]:
        for _j in range(_i * _i, MAX_SUM + 1, _i):
            _is_prime[_j] = False
for _i in range(2, MAX_SUM + 1):
    if _is_prime[_i]:
        _primes.append(_i)

# Precompute all combinations of 3 distinct integers in [1, 128]
# We use a smaller search range initially for speed, but precompute the full set of candidates
# Actually, let's just precompute all possible A and B sets of size 3 from [1, 128]
# That's C(128,3) which is ~340k. That's too many to store in memory efficiently if we do nested loops.
# Better to iterate on the fly but with a smarter strategy.

# Strategy:
# We want to find A, B such that the 9 sums are all distinct and pairwise coprime.
# The easiest way to ensure pairwise coprime is to make all 9 sums prime.
# So we look for A, B such that all a+b in A+B are prime.

# Let's iterate over A and B from a smaller range first, say [1, 50], and if not found, expand.
# But we need to be careful with time.

# Let's try a different approach:
# Iterate over possible sums? No, that's hard.
# Let's stick to the nested loop but optimize the inner check.

def _check_pairwise_coprime(vals):
    """Check if a list of integers is pairwise coprime."""
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Search for all-prime distinct sumset (cardinality 9)
    # We limit the search to small numbers where primes are denser
    # Start with a small range, expand if needed
    
    # Let's try to find a solution quickly.
    # If we find one with 9 distinct primes, we return it immediately as it's optimal.
    
    # We can speed up by checking distinctness first.
    
    # Let's iterate A and B from 1 to 40 first.
    for limit in [20, 30, 40, 50, 100, 128]:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
            
        # Iterate over all combinations of 3 distinct integers from [1, limit]
        # To avoid redundant checks, we can fix A and search for B.
        # Or just nested loops.
        
        # Optimization: Precompute primes for quick lookup.
        # We already have _is_prime.
        
        # Let's use itertools.combinations for A
        for A in itertools.combinations(range(1, limit + 1), 3):
            if time.time() - t_start > TIME_LIMIT - 2:
                break
                
            # For each A, search for B
            for B in itertools.combinations(range(1, limit + 1), 3):
                # Check sums
                sums = [a + b for a in A for b in B]
                
                # Check distinctness
                if len(set(sums)) != 9:
                    continue
                
                # Check if all are prime (stronger condition, implies pairwise coprime)
                # If all are prime, they are pairwise coprime.
                if all(_is_prime[s] for s in sums):
                    return [list(A), list(B)]
                    
                # Fallback: Check pairwise coprime for distinct sums
                # This is slower, so only do it if we are close to the time limit
                # or if we want to be more general. But prime is easier.
                # Let's stick to prime for now as it's the most likely to succeed.
                
    # If we didn't find a prime solution, try a broader search for pairwise coprime
    # This is much slower, so only if we have time.
    if time.time() - t_start < TIME_LIMIT - 10:
        # Try a smaller range for pairwise coprime check
        for limit in [20, 30]:
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for A in itertools.combinations(range(1, limit + 1), 3):
                if time.time() - t_start > TIME_LIMIT - 2:
                    break
                for B in itertools.combinations(range(1, limit + 1), 3):
                    sums = [a + b for a in A for b in B]
                    distinct_sums = list(set(sums))
                    if len(distinct_sums) < 9:
                        continue
                    if _check_pairwise_coprime(distinct_sums):
                        # Found a valid solution
                        # Score is based on cardinality. 9 is max.
                        return [list(A), list(B)]
                        
    return best_result

# EVOLVE-BLOCK-END
