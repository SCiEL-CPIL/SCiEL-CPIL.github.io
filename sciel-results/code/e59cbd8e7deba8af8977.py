import time
import random
import math
from itertools import combinations

# Precompute smallest prime factor up to 20000 (max sum is 9999+10000=19999)
MAX_SUM = 20000
_spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM ** 0.5) + 1):
    if _spf[i] == i:
        for j in range(i * i, MAX_SUM + 1, i):
            if _spf[j] == j:
                _spf[j] = i

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = _spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    prime_union = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            prime_union |= _prime_factors(s)
    return len(prime_union)

def _try_multiples_of(base_arr, max_val, time_limit):
    """Try to improve by using multiples of a small set of primes."""
    start = time.time()
    best = list(base_arr)
    best_score = _count_distinct_primes(best)
    
    # Try to find a set where all pairwise sums are divisible by small primes
    # Strategy: pick a modulus m, and select numbers congruent to specific residues mod m
    # such that sums are divisible by a small set of primes
    
    # Key insight: if all numbers are congruent to 0 mod p, then all sums are 0 mod p
    # But we need distinct numbers. So use multiples of p.
    # If we use multiples of p1, p2, ..., pk, sums are divisible by those primes.
    
    # Better: choose numbers so that all pairwise sums are divisible by a fixed small prime p.
    # This means all numbers must be congruent mod p (since a+b ≡ 0 mod p for all pairs
    # means all a_i ≡ r mod p where 2r ≡ 0 mod p, so r ≡ 0 if p is odd, or r ≡ 0 or p/2 if p even)
    
    # For odd prime p: all numbers ≡ 0 mod p → all sums ≡ 0 mod p
    # For p=2: all numbers ≡ 0 mod 2 → all sums ≡ 0 mod 2 (even)
    
    # So if all numbers are multiples of p, all pairwise sums are divisible by p.
    # To minimize distinct primes, we want ALL pairwise sums to have factors from a tiny set.
    
    # Let's try: all numbers are multiples of 2 → all sums even (factor 2)
    # Then we need to minimize additional primes from (sum/2)
    
    # Try different base moduli
    for base_prime in [2, 3, 5, 7]:
        # Generate candidate sets as multiples of base_prime
        candidates = list(range(base_prime, max_val + 1, base_prime))
        
        if len(candidates) < 12:
            continue
        
        # Greedy: pick 12 numbers from candidates to minimize distinct prime factors of sums
        # Start with first 12 multiples
        current = candidates[:12]
        current_score = _count_distinct_primes(current)
        
        if current_score < best_score:
            best = list(current)
            best_score = current_score
        
        # Local search: try swapping elements
        for _ in range(2000):
            if time.time() - start > time_limit - 5:
                break
            improved = False
            # Random swap
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.choice(candidates)
            if new_val == old_val or new_val in current:
                continue
            current[idx] = new_val
            new_score = _count_distinct_primes(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best = list(current)
                    best_score = current_score
            else:
                current[idx] = old_val
        
        if best_score <= 3:
            break
    
    return best, best_score

def _greedy_construct(max_val, time_limit):
    """Greedy construction: build set to minimize distinct prime factors."""
    start = time.time()
    
    # Strategy: try to make all pairwise sums divisible by a small set of primes
    # If all numbers are multiples of L = lcm of small primes, all sums are divisible by L's primes
    
    # Best case: all sums divisible by just prime 2 → need all numbers same parity
    # Let's try: all even numbers, then minimize primes in (sum/2)
    
    # Phase 1: try all-multiples-of-p approach
    best_arr = list(range(1, 13))
    best_score = _count_distinct_primes(best_arr)
    
    # Try multiples of 2
    evens = list(range(2, max_val + 1, 2))
    for start_idx in [0, 1, 2]:
        if start_idx + 12 > len(evens):
            break
        trial = evens[start_idx:start_idx+12]
        score = _count_distinct_primes(trial)
        if score < best_score:
            best_arr = list(trial)
            best_score = score
    
    # Try multiples of 3
    threes = list(range(3, max_val + 1, 3))
    for start_idx in [0, 1, 2]:
        if start_idx + 12 > len(threes):
            break
        trial = threes[start_idx:start_idx+12]
        score = _count_distinct_primes(trial)
        if score < best_score:
            best_arr = list(trial)
            best_score = score
    
    # Try multiples of 6 (lcm of 2 and 3)
    sixes = list(range(6, max_val + 1, 6))
    for start_idx in [0, 1, 2]:
        if start_idx + 12 > len(sixes):
            break
        trial = sixes[start_idx:start_idx+12]
        score = _count_distinct_primes(trial)
        if score < best_score:
            best_arr = list(trial)
            best_score = score
    
    # Local search on the best arrangement
    if time.time() - start < time_limit - 10:
        best_arr, best_score = _local_search(best_arr, max_val, time_limit - 10, start)
    
    return best_arr

def _local_search(arr, max_val, time_limit, start_time=None):
    """Local search to improve the set."""
    if start_time is None:
        start_time = time.time()
    best = list(arr)
    best_score = _count_distinct_primes(best)
    current = list(best)
    current_score = best_score
    
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        improved = False
        # Try moving one element
        idx = random.randint(0, 11)
        old_val = current[idx]
        # Try random new value
        new_val = random.randint(1, max_val)
        if new_val == old_val or new_val in current:
            continue
        current[idx] = new_val
        new_score = _count_distinct_primes(current)
        if new_score < current_score:
            current_score = new_score
            improved = True
            if new_score < best_score:
                best = list(current)
                best_score = new_score
        else:
            current[idx] = old_val
        
        # Occasionally try structured moves: make element a multiple of small prime
        if not improved and random.random() < 0.3:
            idx = random.randint(0, 11)
            old_val = current[idx]
            small_prime = random.choice([2, 3, 5, 7])
            new_val = random.randint(1, max_val // small_prime) * small_prime
            if new_val == old_val or new_val in current or new_val > max_val or new_val < 1:
                current[idx] = old_val
                continue
            current[idx] = new_val
            new_score = _count_distinct_primes(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best = list(current)
                    best_score = new_score
            else:
                current[idx] = old_val
    
    return best, best_score

def construct():
    start = time.time()
    DEADLINE = 75  # Leave 15s margin for evaluation overhead
    
    # Phase 1: Quick greedy construction
    arr = _greedy_construct(10000, 20)
    score = _count_distinct_primes(arr)
    
    # Phase 2: More thorough search with multiple strategies
    if time.time() - start < DEADLINE - 5:
        best = list(arr)
        best_score = score
        
        # Try multiples-of-p approach more thoroughly
        multi_start = time.time()
        multi_arr, multi_score = _try_multiples_of(arr, 10000, min(DEADLINE - (time.time() - start), 30))
        if multi_score < best_score:
            best = multi_arr
            best_score = multi_score
        
        # Phase 3: Local search with more iterations
        if time.time() - start < DEADLINE - 3:
            remaining = DEADLINE - (time.time() - start) - 3
            if remaining > 5:
                ls_arr, ls_score = _local_search(best, 10000, remaining, start)
                if ls_score < best_score:
                    best = ls_arr
                    best_score = ls_score
    
    # Final check: ensure validity
    assert len(best) == 12
    assert len(set(best)) == 12
    assert all(1 <= x <= 10000 for x in best)
    
    return best
