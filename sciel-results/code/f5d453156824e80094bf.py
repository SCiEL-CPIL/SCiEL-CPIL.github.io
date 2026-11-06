"""
Optimized smooth two-summand decomposition.
Combines the LPF sieve approach (for guaranteed global optimum) with
incremental smooth number generation (for early termination when a solution
is found at a small prime bound).

Strategy:
1. Use incremental smooth number generation: start with primes in increasing order,
   maintain the set of all numbers <= TARGET whose prime factors are from the primes
   seen so far. After adding each prime p, check if any a in the smooth set has
   TARGET - a also in the smooth set. If yes, return immediately (this is optimal
   since we iterate primes in increasing order).
2. If no solution is found with small primes (unlikely for this target), fall back
   to a full LPF sieve and linear scan to guarantee correctness.
"""
import time
import math
from collections import deque

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    TARGET = 123457

    # Sieve of Eratosthenes to get primes up to a reasonable limit
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Generate primes up to 1000 (more than enough; the answer's max prime factor is small)
    primes = sieve(1000)

    # Incremental smooth number generation
    # smooth_set contains all numbers <= TARGET whose prime factors are from primes processed so far
    smooth_set = {1}

    # Process each prime incrementally
    for i, p in enumerate(primes):
        if time.time() > deadline:
            break

        # Add all new smooth numbers involving prime p
        # For each existing smooth number x, add x*p, x*p^2, ... <= TARGET
        # We need to iterate over a snapshot since we're modifying the set
        existing = list(smooth_set)
        for x in existing:
            nx = x * p
            while nx <= TARGET:
                if nx not in smooth_set:
                    smooth_set.add(nx)
                nx *= p

        # Check for pairs: is there an a in smooth_set such that TARGET - a is also in smooth_set?
        found_pair = None
        for a in smooth_set:
            if a > TARGET:
                continue
            b = TARGET - a
            if b <= 0:
                continue
            if b in smooth_set:
                # Found a valid pair with max prime factor <= p
                # Since we iterate primes in increasing order and no solution was found earlier,
                # the max prime factor is exactly p (or possibly less, but that would have been caught)
                found_pair = (a, b)
                break

        if found_pair:
            # This is the optimal solution: smallest possible max prime factor
            return [found_pair[0], found_pair[1]]

    # Fallback: full LPF sieve and linear scan (should not be reached for this target)
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    best_val = float('inf')
    best_pair = [1, TARGET - 1]
    half = TARGET // 2
    for a in range(1, half + 1):
        b = TARGET - a
        va = lpf[a]
        vb = lpf[b]
        val = va if va > vb else vb
        if val < best_val:
            best_val = val
            best_pair = [a, b]
        if time.time() > deadline:
            break

    return best_pair
