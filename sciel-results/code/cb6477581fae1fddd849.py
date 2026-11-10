# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    MAX_N = 10000000
    
    # Build SPF (Smallest Prime Factor) sieve up to MAX_N
    # This allows O(log n) sigma computation for any n <= MAX_N
    spf = list(range(MAX_N + 1))
    spf[0] = 0
    spf[1] = 1
    
    # Sieve of Eratosthenes variant to build SPF
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Cache for sigma values
    sigma_cache = {1: 1}
    
    def sigma(n):
        """Compute sigma(n) using SPF, with caching."""
        if n in sigma_cache:
            return sigma_cache[n]
        
        if time.time() > deadline - 1:
            return -1  # Signal timeout
        
        orig = n
        result = 1
        while n > 1:
            p = spf[n]
            # Compute p^a * (1 + p + ... + p^a)
            power = 1
            term = 1
            while n % p == 0:
                n //= p
                power *= p
                term += power
            result *= term
            if result > 10**18:  # Overflow guard (sigma values can be large)
                break
        
        sigma_cache[orig] = result
        return result
    
    # Strategy: Search over k = 2a - 3b (the numerator of the error when expressed as k/(2b))
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b) = |k| / (2b)
    # For each k (starting from small |k|), we want to find large b such that:
    #   a = (3b + k) / 2 is a positive integer
    #   1 <= a, b <= MAX_N
    #   sigma(a) == sigma(b)
    
    best_a = 11
    best_b = 6
    best_err = Fraction(1, 3)  # |11/6 - 3/2| = |22/12 - 18/12| = 4/12 = 1/3
    
    # For each k, we need 3b + k to be even, so b must have the same parity as k (since 3b has same parity as b)
    # Actually: 3b + k must be even. 3b has same parity as b. So b + k must be even, i.e., b and k have same parity.
    
    # We iterate k in order of increasing |k|, and for each k, try large b values.
    # For a given k, b must satisfy:
    #   a = (3b + k) / 2
    #   1 <= a <= MAX_N => 2 <= 3b + k <= 2*MAX_N => (2-k)/3 <= b <= (2*MAX_N - k)/3
    #   1 <= b <= MAX_N
    #   b and k have same parity
    
    # We want to maximize b (to minimize k/(2b)) for each k.
    # So for each k, we try b from the maximum possible downward.
    
    # How many b values to try per k? 
    # If we try too many, we spend too much time. Let's try a limited number.
    # The key insight: for small k, even a single match with large b gives a very small error.
    # Let's try up to ~200 candidates per k for small k, fewer for larger k.
    
    max_k_to_check = 200  # Check k from -max_k to max_k
    checks_per_k_small = 500  # For |k| <= 10
    checks_per_k_medium = 100  # For 10 < |k| <= 50
    checks_per_k_large = 20  # For |k| > 50
    
    for k in range(1, max_k_to_check + 1):
        if time.time() > deadline - 2:
            break
        
        # Try both k and -k
        for kk in [k, -k]:
            if time.time() > deadline - 1:
                break
            
            # Determine parity constraint: b must have same parity as kk
            # a = (3b + kk) / 2
            # b range: max(1, ceil((2 - kk) / 3)) to min(MAX_N, floor((2*MAX_N - kk) / 3))
            
            b_min = max(1, math.ceil((2 - kk) / 3))
            b_max = min(MAX_N, math.floor((2 * MAX_N - kk) / 3))
            
            if b_min > b_max:
                continue
            
            # Adjust b_max to correct parity
            if b_max % 2 != kk % 2:
                b_max -= 1
            
            if b_min > b_max:
                continue
            
            # Determine how many checks to do
            if k <= 10:
                num_checks = checks_per_k_small
            elif k <= 50:
                num_checks = checks_per_k_medium
            else:
                num_checks = checks_per_k_large
            
            # Try b values from b_max downward
            count = 0
            b = b_max
            while b >= b_min and count < num_checks:
                if time.time() > deadline - 1:
                    break
                
                a = (3 * b + kk) // 2
                # Verify that a is indeed an integer (it should be by construction)
                if 3 * b + kk != 2 * a:
                    b -= 2 if kk % 2 == 1 else 2
                    # Actually, step should be 2 to maintain parity
                    count += 1
                    continue
                
                if 1 <= a <= MAX_N and a != b:
                    sa = sigma(a)
                    sb = sigma(b)
                    if sa != -1 and sb != -1 and sa == sb:
                        err = Fraction(abs(kk), 2 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            # If we found k=1 with very large b, that's excellent
                            if kk == 1 and b > MAX_N // 2:
                                # Very good result, can potentially stop early
                                pass
                
                # Move to next b with same parity
                if kk % 2 == 0:
                    b -= 2
                else:
                    b -= 2
                count += 1
        
        # Early termination: if best_err is already very small, we might not improve much
        # The minimum possible error for a given b is 1/(2b) (when k=1)
        # If best_err < 1/(2 * MAX_N), we can't do better
        if best_err < Fraction(1, 2 * MAX_N):
            break
    
    # Additional search: try to find pairs where a/b is close to 3/2 by scanning
    # specific arithmetic progressions
    # For ratio 3/2, a = 3t, b = 2t gives exact ratio but sigma(3t) != sigma(2t) generally
    # Let's also try nearby values
    
    if time.time() < deadline - 5:
        # Try a few more specific patterns
        # Look for b where 3b/2 is close to an integer, and check sigma match
        for t in range(MAX_N // 2, max(MAX_N // 2 - 5000, 0), -1):
            if time.time() > deadline - 2:
                break
            b = 2 * t
            a = 3 * t
            if a <= MAX_N:
                sa = sigma(a)
                sb = sigma(b)
                if sa != -1 and sb != -1 and sa == sb:
                    err = Fraction(0, 1)
                    # Error is 0, which is invalid per task
                    # But let's check nearby
                    pass
            
            # Check a = 3t ± 1, b = 2t
            for da in [-2, -1, 1, 2]:
                a_test = 3 * t + da
                if 1 <= a_test <= MAX_N and a_test != b:
                    sa = sigma(a_test)
                    sb = sigma(b)
                    if sa != -1 and sb != -1 and sa == sb:
                        err = Fraction(abs(da), 2 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a_test
                            best_b = b
    
    # Ensure we have a valid fallback
    if best_err >= Fraction(1, 2):
        best_a = 11
        best_b = 6
        best_err = Fraction(1, 3)
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
