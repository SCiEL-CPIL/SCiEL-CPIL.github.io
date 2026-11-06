"""Program to minimize distinct prime factors across pairwise sums."""
import random
import math
from itertools import combinations

def _sieve_smallest_prime(limit):
    """Return list spf where spf[i] is smallest prime factor of i."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors(n, spf):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr, spf):
    """Count distinct primes dividing all pairwise sums of arr."""
    prime_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                prime_set.add(p)
                while n % p == 0:
                    n //= p
    return len(prime_set)

def construct():
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor up to 20000 (max sum of two <= 10000)
    limit = 2 * max_val
    spf = _sieve_smallest_prime(limit)
    
    # Initial strategy: try to use numbers with few distinct prime factors in their sums
    # A good starting point: use numbers that are multiples of small primes
    # Let's start with powers of 2 and 3 mixed, or just consecutive even numbers
    
    # Try a few initial configurations
    candidates = [
        list(range(1, n + 1)),  # 1..12
        list(range(2, 2 * n + 1, 2)),  # 2,4,6,...,24
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    
    best_score = float('inf')
    best_arr = None
    
    for cand in candidates:
        if len(set(cand)) != n or any(x < 1 or x > max_val for x in cand):
            continue
        score = _count_distinct_primes(cand, spf)
        if score < best_score:
            best_score = score
            best_arr = list(cand)
    
    if best_arr is None:
        best_arr = list(range(1, n + 1))
        best_score = _count_distinct_primes(best_arr, spf)
    
    # Local search: try to replace elements to reduce the number of distinct primes
    random.seed(42)
    improved = True
    max_iterations = 50000
    
    iter_count = 0
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # Try all possible swaps/replacements
        for i in range(n):
            # Try replacing arr[i] with another value
            # Generate candidate values: near current, or values that share prime factors
            old_val = best_arr[i]
            
            # Generate candidate replacements
            candidates_repl = set()
            
            # Neighbors
            for delta in range(-50, 51):
                v = old_val + delta
                if 1 <= v <= max_val and v != old_val:
                    candidates_repl.add(v)
            
            # Small primes multiples
            for p in [2, 3, 5, 7, 11, 13]:
                for mult in range(1, max_val // p + 1, 7):  # sample sparsely
                    v = p * mult
                    if 1 <= v <= max_val and v != old_val:
                        candidates_repl.add(v)
            
            # Powers of small numbers
            for base in [2, 3, 4, 5]:
                v = 1
                while v <= max_val:
                    if v != old_val:
                        candidates_repl.add(v)
                    v *= base
            
            if not candidates_repl:
                continue
            
            # Sort by some heuristic: prefer values that share factors with existing sums
            for new_val in sorted(candidates_repl, key=lambda x: abs(x - old_val)):
                if new_val in best_arr:  # Must keep distinct
                    continue
                
                trial = list(best_arr)
                trial[i] = new_val
                
                new_score = _count_distinct_primes(trial, spf)
                if new_score < best_score:
                    best_score = new_score
                    best_arr = trial
                    improved = True
                    break  # Restart inner loop with new best
    
    return best_arr
