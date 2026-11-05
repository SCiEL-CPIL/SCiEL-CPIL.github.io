import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    Uses a direct scan to find the optimal pair.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds
    
    # Sieve of Eratosthenes to find all primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    # Precompute largest prime factor for all numbers up to TARGET
    # lp[n] will store the largest prime factor of n
    lp = [0] * (TARGET + 1)
    lp[1] = 1  # P(1) = 1
    
    # We can fill lp by iterating through primes
    # A number's largest prime factor is the largest prime that divides it.
    # We can iterate p from 2 to TARGET. If p is prime, update lp for all multiples.
    # This is O(N log log N) roughly, similar to sieve.
    for p in range(2, TARGET + 1):
        if is_prime[p]:
            for multiple in range(p, TARGET + 1, p):
                lp[multiple] = p

    # Single pass to find the pair (a, b) with a+b=TARGET
    # minimizing max(lp[a], lp[b])
    best_a = 1
    best_b = TARGET - 1
    best_val = max(lp[1], lp[TARGET - 1])
    
    # We only need to check a up to TARGET // 2 because the pair (a, b) is symmetric
    # However, lp[a] and lp[b] might differ, so we should check all a?
    # Actually, max(lp[a], lp[b]) == max(lp[b], lp[a]).
    # So checking a in 1..TARGET//2 is sufficient to cover all unique pairs {a, b}.
    for a in range(2, TARGET // 2 + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        b = TARGET - a
        current_val = max(lp[a], lp[b])
        if current_val < best_val:
            best_val = current_val
            best_a = a
            best_b = b
    
    # Check the middle if TARGET is odd? TARGET is odd (123457).
    # a = (TARGET-1)/2 and b = (TARGET+1)/2 are distinct.
    # My loop goes up to TARGET//2. 
    # If TARGET is odd, a ranges 1 to (TARGET-1)/2.
    # This covers all pairs where a < b.
    # What if the optimal pair has a > b? It's the same pair.
    # So checking a < b is sufficient.
    
    # Just in case, check a = TARGET//2 + 1? No, that's symmetric to a = TARGET//2 - 1.
    # Since TARGET is odd, a and b are never equal.
    # The set of pairs {a, TARGET-a} for a in 1..(TARGET-1)/2 is the set of all distinct unordered pairs.
    
    return [best_a, best_b]
