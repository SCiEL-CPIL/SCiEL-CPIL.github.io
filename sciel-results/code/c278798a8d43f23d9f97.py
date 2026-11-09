"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _largest_prime_factor(n):
    """Compute the largest prime factor of n."""
    if n <= 1:
        return 1
    p = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            p = d
            n //= d
        d += 1
    if n > 1:
        p = n
    return p

def _gen_smooth_numbers(max_val, primes):
    """Generate all numbers <= max_val with prime factors only from 'primes'."""
    smooth = set()
    def backtrack(idx, current):
        if current > max_val:
            return
        if current > 1:
            smooth.add(current)
        for i in range(idx, len(primes)):
            p = primes[i]
            if current * p > max_val:
                break
            backtrack(i, current * p)
    backtrack(0, 1)
    return smooth

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    N = 123457
    
    # Try increasing smoothness bounds
    for bound in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        
        # Get all primes up to bound
        primes = []
        for p in range(2, bound + 1):
            is_prime = True
            for q in range(2, isqrt(p) + 1):
                if p % q == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(p)
        
        smooth = _gen_smooth_numbers(N, primes)
        smooth_list = sorted(smooth)
        
        # Two-pointer to find a + b = N
        best_pair = None
        best_max_pf = None
        
        i, j = 0, len(smooth_list) - 1
        while i < j:
            s = smooth_list[i] + smooth_list[j]
            if s == N:
                # Found a pair; both are smooth with primes <= bound
                pf_a = _largest_prime_factor(smooth_list[i])
                pf_b = _largest_prime_factor(smooth_list[j])
                max_pf = max(pf_a, pf_b)
                if best_max_pf is None or max_pf < best_max_pf:
                    best_max_pf = max_pf
                    best_pair = (smooth_list[i], smooth_list[j])
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        
        if best_pair is not None:
            # Verify
            a, b = best_pair
            assert a + b == N and a >= 1 and b >= 1
            return [a, b]
    
    # Fallback: exhaustive search over all a in [1, N-1]
    if time.time() > deadline:
        # Return a reasonable default
        return [1, N - 1]
    
    best_a = 1
    best_val = float('inf')
    for a in range(1, N):
        b = N - a
        pa = _largest_prime_factor(a)
        pb = _largest_prime_factor(b)
        v = max(pa, pb)
        if v < best_val:
            best_val = v
            best_a = a
        if time.time() > deadline:
            break
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
