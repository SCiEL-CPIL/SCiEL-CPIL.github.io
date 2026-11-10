"""Solve: find a,b positive with a+b=123457 minimizing max(P(a),P(b))."""
import math
import time

TARGET = 123457
DEADLINE = 85.0  # safety margin before 90s kill


def _largest_prime_factor_sieve(n):
    """Compute largest prime factor for all integers 1..n using sieve."""
    lpf = list(range(n + 1))
    for i in range(2, int(math.isqrt(n)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                lpf[j] = i
    # For numbers that are prime, lpf[x] = x already
    # For 1, lpf[1] = 1
    return lpf


def construct():
    start_time = time.time()
    
    # Precompute largest prime factor for all numbers up to TARGET
    lpf = _largest_prime_factor_sieve(TARGET)
    
    # P(1) = 1 by definition; lpf[1] is already 1
    # P(prime p) = p; lpf[p] = p for primes
    
    best_score = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    # Iterate all splits a + b = TARGET, a from 1 to TARGET//2 (symmetry)
    half = TARGET // 2
    for a in range(1, half + 1):
        b = TARGET - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            # Check deadline
            if time.time() - start_time > DEADLINE:
                break
        # Early exit: if best_score is 2, that's the minimum possible (both must be even, but sum is odd so one is odd -> P(odd)>=3... actually P(1)=1, so min could be 3)
        # Actually min possible: one number is 1 (P=1), other is 123456. P(123456) is large.
        # The theoretical minimum for max(P(a),P(b)) where a+b is odd: one is even, one is odd.
        # P(even) >= 2, P(odd) >= 3 (unless odd=1, P=1). So min max is at least 3.
        if best_score <= 3:
            break
    
    return [best_a, best_b]
