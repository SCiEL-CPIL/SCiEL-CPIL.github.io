"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _score_set(arr):
    """Count distinct prime factors across all pairwise sums."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            prime_set.update(_prime_factors(s))
    return len(prime_set)

def _prime_factors_fast(n, smallest_prime):
    """Use precomputed smallest prime factor for speed."""
    factors = set()
    if n <= 1:
        return factors
    while n > 1:
        p = smallest_prime[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score_set_fast(arr, smallest_prime):
    """Fast scoring using precomputed smallest prime factor."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i+1, n):
            s = ai + arr[j]
            if s > 1:
                m = s
                while m > 1:
                    p = smallest_prime[m]
                    prime_set.add(p)
                    while m % p == 0:
                        m //= p
    return len(prime_set)

def _score_reduced(arr_reduced, smallest_prime):
    """Score the reduced problem: count distinct prime factors of pairwise sums of arr_reduced.
    arr_reduced contains values such that original = arr_reduced[i] * d."""
    prime_set = set()
    n = len(arr_reduced)
    for i in range(n):
        ai = arr_reduced[i]
        for j in range(i+1, n):
            s = ai + arr_reduced[j]
            if s > 1:
                m = s
                while m > 1:
                    p = smallest_prime[m]
                    prime_set.add(p)
                    while m % p == 0:
                        m //= p
    return len(prime_set)

def _total_score(arr, d, smallest_prime):
    """Total distinct prime factors = primes of d + primes of reduced pairwise sums."""
    # Primes of d
    d_primes = set()
    m = d
    while m > 1:
        p = smallest_prime[m]
        d_primes.add(p)
        while m % p == 0:
            m //= p
    # Primes of reduced sums
    reduced = [x // d for x in arr]
    sum_primes = set()
    n = len(reduced)
    for i in range(n):
        ai = reduced[i]
        for j in range(i+1, n):
            s = ai + reduced[j]
            if s > 1:
                m2 = s
                while m2 > 1:
                    p = smallest_prime[m2]
                    sum_primes.add(p)
                    while m2 % p == 0:
                        m2 //= p
    return len(d_primes | sum_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    # Precompute smallest prime factor up to 20000
    MAX_VAL = 20000
    smallest_prime = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if smallest_prime[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if smallest_prime[j] == j:
                    smallest_prime[j] = i
    
    random.seed(42)
    
    best_set = None
    best_score = float('inf')
    
    # Key insight: if all numbers are multiples of d, then all pairwise sums
    # are multiples of d. The total distinct primes = primes(d) + primes of
    # (a_i/d + a_j/d) for all pairs.
    # We want to maximize d (to absorb many primes) while keeping the
    # reduced problem simple.
    
    # Try various divisors d. For each d, we need 12 distinct multiples of d
    # up to 10000, so d <= 10000/12 ≈ 833.
    # But larger d means fewer choices for the reduced values.
    # The reduced values r_i must be distinct positive integers, and
    # r_i * d <= 10000, so r_i <= 10000 // d.
    
    # Strategy: for each promising d, find 12 distinct reduced values
    # that minimize the number of distinct prime factors of their pairwise sums.
    
    # Promising d values: those with few prime factors themselves,
    # and large enough to constrain the reduced problem.
    # Actually, we want d to have FEW prime factors (they're always included)
    # but the reduced sums should have FEW additional primes.
    
    # Let's try d values and for each, do local search on reduced values.
    
    # Candidate d values: products of small primes, up to ~800
    candidate_ds = []
    for d in range(2, 834):
        # Only consider d where we can fit 12 distinct multiples
        if 12 * d <= 10000:
            candidate_ds.append(d)
    
    # Sort by number of prime factors of d (prefer fewer)
    def num_prime_factors(d):
        m = d
        cnt = set()
        while m > 1:
            p = smallest_prime[m]
            cnt.add(p)
            while m % p == 0:
                m //= p
        return len(cnt)
    
    candidate_ds.sort(key=num_prime_factors)
    
    # Try top candidates for d
    for d in candidate_ds[:200]:
        if time.time() > deadline:
            break
        
        max_reduced = 10000 // d
        if max_reduced < 12:
            continue
        
        # Initial: smallest 12 reduced values
        initial_reduced = list(range(1, 13))
        if initial_reduced[-1] > max_reduced:
            continue
        
        # Score this initial
        cur_score = _score_reduced(initial_reduced, smallest_prime) + num_prime_factors(d)
        if cur_score < best_score:
            best_score = cur_score
            best_set = [x * d for x in initial_reduced]
        
        # Local search on reduced values
        improved = True
        local_best = initial_reduced[:]
        local_score = cur_score
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = local_best[i]
                # Try nearby values
                for new_r in range(max(1, old - 10), min(max_reduced + 1, old + 11)):
                    if new_r == old:
                        continue
                    if new_r in local_best:
                        continue
                    test = local_best[:]
                    test[i] = new_r
                    test.sort()
                    ns = _score_reduced(test, smallest_prime) + num_prime_factors(d)
                    if ns < local_score:
                        local_score = ns
                        local_best = test
                        improved = True
                        break
        
        if local_score < best_score:
            best_score = local_score
            best_set = [x * d for x in local_best]
    
    # Also try direct local search without the d structure
    if time.time() < deadline:
        # Try even numbers
        candidates_even = list(range(2, 10000, 2))
        initial_even = candidates_even[:12]
        score_even = _score_set_fast(initial_even, smallest_prime)
        if score_even < best_score:
            best_score = score_even
            best_set = initial_even[:]
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = best_set[i]
                for offset in range(-20, 21):
                    if offset == 0:
                        continue
                    new_val = old + offset
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in best_set:
                        continue
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    ns = _score_set_fast(test_set, smallest_prime)
                    if ns < best_score:
                        best_score = ns
                        best_set = test_set
                        improved = True
                        break
    
    return best_set

# EVOLVE-BLOCK-END
