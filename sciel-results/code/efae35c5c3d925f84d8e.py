"""Sieve-based search for n minimizing the largest prime factor of n(n+1)."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    LO = 10000
    HI = 1000000  # inclusive upper bound for n
    
    # We need largest prime factor for numbers in [LO, HI+1]
    # Strategy: compute LPF for all numbers up to HI+1 using a sieve
    # Then for each n in [LO, HI], compute max(lpf[n], lpf[n+1]) and find min
    
    max_val = HI + 1  # we need LPF up to 1000001
    
    # Sieve for largest prime factor
    lpf = list(range(max_val + 1))  # lpf[i] will hold the largest prime factor of i
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve of Eratosthenes variant: for each prime p, update lpf for multiples
    # This is O(N log log N)
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i  # keep overwriting with larger primes
    
    # For numbers that are prime themselves, lpf[p] = p (already set)
    # For composite numbers, lpf[n] is the largest prime that divided it in the sieve
    
    # Now find n in [LO, HI] that minimizes max(lpf[n], lpf[n+1])
    best_n = LO
    best_val = max(lpf[LO], lpf[LO + 1])
    
    # Vectorized search using a loop (NumPy not strictly needed but let's use it for speed)
    # Actually, let's just do a fast Python loop since we're just scanning an array
    for n in range(LO + 1, HI + 1):
        v = lpf[n]
        if v < lpf[n + 1]:
            v = lpf[n + 1]
        if v < best_val:
            best_val = v
            best_n = n
        # Check deadline
        if n % 100000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
