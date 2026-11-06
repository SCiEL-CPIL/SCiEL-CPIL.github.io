"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

# Precompute prime factors for all possible sums (1 to 20000)
_MAX_SUM = 20000
_prime_factors_cache = [None] * (_MAX_SUM + 1)

def _init_factors():
    """Precompute prime factors for all numbers up to _MAX_SUM."""
    # Sieve of Eratosthenes variant to find smallest prime factor, then derive distinct factors
    spf = list(range(_MAX_SUM + 1))
    for i in range(2, int(_MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, _MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for n in range(2, _MAX_SUM + 1):
        factors = set()
        x = n
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        _prime_factors_cache[n] = factors
    
    # Handle 1 (no prime factors)
    _prime_factors_cache[1] = set()

_init_factors()

def _get_factors(n):
    return _prime_factors_cache[n]

def _find_clique_for_prime_set(prime_set, time_limit=5.0):
    """
    Finds a clique of size 12 in the graph where vertices are numbers 1..10000
    and edges exist if sum of endpoints has prime factors subset of prime_set.
    """
    target_size = 12
    max_val = 10000
    start_time = time.time()
    
    # Generate S-smooth numbers up to 20000
    smooth_numbers = set()
    def _gen_smooth(primes, limit, current, idx=0):
        if current > limit:
            return
        if idx == len(primes):
            if current > 1:
                smooth_numbers.add(current)
            return
        mult = 1
        while current * mult <= limit:
            _gen_smooth(primes, limit, current * mult, idx + 1)
            mult *= primes[idx]
            
    _gen_smooth(sorted(prime_set), 20000, 1)
    smooth_set = smooth_numbers
    smooth_list_sorted = sorted(smooth_numbers)
    
    # Backtracking search
    def backtrack(clique, min_val):
        if len(clique) == target_size:
            return clique
        
        if time.time() - start_time > time_limit:
            return None
            
        if clique:
            last = clique[-1]
            # Generate candidates from last
            cands = []
            for s in smooth_list_sorted:
                y = s - last
                if y > last and y <= max_val:
                    # Check against all previous
                    ok = True
                    for x in clique[:-1]:
                        if (x + y) not in smooth_set:
                            ok = False
                            break
                    if ok:
                        cands.append(y)
            
            # Try candidates
            for y in sorted(cands):
                res = backtrack(clique + [y], y + 1)
                if res:
                    return res
            return None
        else:
            # First element
            # Try small numbers first
            for y in range(min_val, max_val + 1):
                res = backtrack([y], y + 1)
                if res:
                    return res
            return None
            
    return backtrack([], 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try small prime sets
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 5, 7, 11},
    ]
    
    best_arr = None
    best_score = float('inf')
    
    for ps in prime_sets_to_try:
        if time.time() > deadline - 5.0:
            break
            
        # Allow less time for each attempt as we go? 
        # Or just a fixed small time per attempt.
        # If a solution is found, it's likely fast.
        # If not, we might waste time.
        # Let's give 5s per attempt.
        
        clique = _find_clique_for_prime_set(ps, time_limit=5.0)
        
        if clique:
            # Calculate actual score
            primes_in_sums = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    s = clique[i] + clique[j]
                    primes_in_sums.update(_get_factors(s))
            
            score = len(primes_in_sums)
            if score < best_score:
                best_score = score
                best_arr = clique
            if score <= 2:
                break
                
    if best_arr is None:
        # Fallback
        # Try a larger set
        ps = {2, 3, 5, 7, 11, 13, 17}
        clique = _find_clique_for_prime_set(ps, time_limit=5.0)
        if clique:
            best_arr = clique
        else:
            # Absolute fallback
            best_arr = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
            
    return best_arr
# EVOLVE-BLOCK-END
