"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _largest_prime_factor(n):
    if n <= 1:
        return 1
    largest = 1
    # check for factor 2
    while n % 2 == 0:
        largest = 2
        n //= 2
    # check for odd factors
    f = 3
    while f * f <= n:
        while n % f == 0:
            largest = f
            n //= f
        f += 2
    if n > 1:
        largest = n
    return largest

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = 123457
    
    # We want to minimize max(P(a), P(b)) where a + b = target.
    # Strategy: Iterate through candidate values for a.
    # For each a, compute b = target - a.
    # Compute P(a) and P(b).
    # Track the minimum of max(P(a), P(b)).
    
    # To be efficient, we can start from the middle and move outwards,
    # or start from small numbers and move up.
    # Since P(n) tends to be large for primes, we want a and b to be composite
    # with small prime factors.
    
    # Let's iterate a from 1 to target//2.
    # To speed up, we can prioritize a and b that are divisible by small primes.
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Heuristic: Check numbers that are divisible by small primes first?
    # Or just iterate? 123457 is small enough that brute force might be okay
    # if the factorization is fast.
    # sqrt(123457) is approx 351.
    # Trial division up to 351 is very fast.
    # 123457 / 2 = 61728 iterations.
    # Each iteration involves factoring a and b.
    # Factoring a number < 123457 takes at most ~350 divisions.
    # 61728 * 350 * 2 ~ 43 million operations.
    # In Python, this might take a few seconds. 85s is plenty.
    
    # Optimization: If P(a) > current best_max_pf, we can skip computing P(b) 
    # if we know P(b) >= 1, but max(P(a), P(b)) >= P(a) > best, so it won't improve.
    # Actually, we need to check if max(P(a), P(b)) < best_max_pf.
    # So if P(a) >= best_max_pf, we can skip? No, because P(b) could be small, 
    # but max would still be >= P(a) >= best_max_pf. So yes, if P(a) >= best_max_pf, 
    # we can skip the rest for this a.
    # Same for b.
    
    for a in range(1, target // 2 + 1):
        if time.time() > deadline:
            break
        
        b = target - a
        
        pa = _largest_prime_factor(a)
        if pa >= best_max_pf:
            continue
            
        pb = _largest_prime_factor(b)
        if pb >= best_max_pf:
            continue
            
        current_max = max(pa, pb)
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            
            # Early exit if we reach the theoretical minimum?
            # The minimum possible value for P(n) is 2 (for even numbers > 1).
            # If we find a pair where both are powers of 2, max is 2.
            # But 2^k + 2^m = 123457 (odd). One must be odd, so P(odd) >= 3.
            # So min possible max is at least 3.
            # If best_max_pf == 3, we can stop?
            # P(n)=3 means n is of form 3^k.
            # 3^a + 3^b = 123457.
            # 3^a (1 + 3^(b-a)) = 123457.
            # 123457 is not divisible by 3 (1+2+3+4+5+7 = 22).
            # So no solution with max=3.
            # Next is 5.
            # If best_max_pf == 5, check if possible.
            # We'll just let the loop run or break if we find a very low value.
            if best_max_pf <= 5:
                # Very unlikely to get lower than 5 for this specific number easily,
                # but let's not break too early unless we are sure.
                # Actually, let's just continue to be safe, or break if < 10.
                pass

    return [best_a, best_b]
# EVOLVE-BLOCK-END
