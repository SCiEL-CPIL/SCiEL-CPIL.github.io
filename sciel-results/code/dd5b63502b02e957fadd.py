"""Rewritten program: minimize distinct prime factors across pairwise sums of 12 distinct ints <=10000."""
import time
import random
import itertools
import math

# Deadline guard
_START = time.time()
_DEADLINE = 85.0  # seconds (leave margin under 90s)

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _factor(n, primes):
    """Return set of prime factors of n using precomputed primes list."""
    factors = set()
    if n <= 1:
        return factors
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(arr, primes):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    n = len(arr)
    all_primes = set()
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            all_primes |= _factor(s, primes)
    return len(all_primes)

def _prime_factors_set(n, primes):
    """Return set of prime factors of n."""
    return _factor(n, primes)

def construct():
    global _START
    _START = time.time()
    _DEADLINE = _START + 85.0
    
    # Precompute primes up to 20000 (max sum is 20000)
    limit = 20000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Known good: powers of 2 give ~9 distinct primes
    # 2^0..2^11: sums are 2^i(1+2^(j-i)), odd factors divide 2^k+1 for k=1..11
    # Let's verify: use exponents 0..11
    powers = [1 << i for i in range(12)]
    best = list(powers)
    best_score = _count_distinct_primes(best, primes)
    
    # Try to improve with local search
    # Strategy: try swapping elements or adjusting values
    
    random.seed(42)
    
    def _evaluate(arr):
        return _count_distinct_primes(arr, primes)
    
    # Local search: try replacing one element with a nearby value
    improved = True
    while improved and _time_left() > 1.0:
        improved = False
        for i in range(len(best)):
            if _time_left() < 1.0:
                break
            # Try nearby values
            orig = best[i]
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in best:
                    continue
                trial = list(best)
                trial[i] = new_val
                score = _evaluate(trial)
                if score < best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break
    
    # Try a different starting point: all multiples of a small number
    # If all numbers are multiples of m, sums are multiples of m, reducing prime variety
    # But we need distinct values
    
    # Try: all numbers = k * base for various base
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if _time_left() < 2.0:
            break
        # Pick 12 distinct multiples
        candidates = [base * i for i in range(1, 100)]
        candidates = [c for c in candidates if c <= 10000]
        if len(candidates) < 12:
            continue
        # Try first 12
        trial = candidates[:12]
        score = _evaluate(trial)
        if score < best_score:
            best = list(trial)
            best_score = score
            improved = True
    
    # Try: numbers of form a * 2^i for fixed a
    for a in range(1, 100):
        if _time_left() < 1.0:
            break
        trial = [a * (1 << i) for i in range(12)]
        if max(trial) > 10000:
            break
        if len(set(trial)) != 12:
            continue
        score = _evaluate(trial)
        if score < best_score:
            best = list(trial)
            best_score = score
    
    # Local search from this new best
    improved = True
    while improved and _time_left() > 1.0:
        improved = False
        for i in range(len(best)):
            if _time_left() < 1.0:
                break
            orig = best[i]
            for delta in range(-100, 101):
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in best:
                    continue
                trial = list(best)
                trial[i] = new_val
                score = _evaluate(trial)
                if score < best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break
    
    # Try random restarts with local search
    for _ in range(50):
        if _time_left() < 2.0:
            break
        # Generate random set biased toward small numbers
        trial = sorted(random.sample(range(1, 200), 12))
        score = _evaluate(trial)
        if score < best_score:
            best = list(trial)
            best_score = score
            # Local search
            improved = True
            while improved and _time_left() > 0.5:
                improved = False
                for i in range(len(best)):
                    if _time_left() < 0.5:
                        break
                    orig = best[i]
                    for delta in range(-50, 51):
                        if delta == 0:
                            continue
                        new_val = orig + delta
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in best:
                            continue
                        t2 = list(best)
                        t2[i] = new_val
                        s2 = _evaluate(t2)
                        if s2 < best_score:
                            best = t2
                            best_score = s2
                            improved = True
                            break
    
    return best
