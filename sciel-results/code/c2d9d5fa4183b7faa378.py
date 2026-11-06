import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 1000001
    # Initialize largest prime factor array with 0
    lpf = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for largest prime factor using numpy vectorization
    # i is the current candidate for prime
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            # Update all multiples of i
            lpf[i:N+1:i] = i
            
            # Check time periodically
            if i % 10000 == 0 and time.time() > deadline:
                break
    
    # If sieve was interrupted, some lpf values might be incomplete.
    # For N=10^6, this should complete in <1s, but handle gracefully.
    # If interrupted, we need to verify completeness or fall back.
    # Check if any value > 1 has lpf == 0 (shouldn't happen if complete)
    # Actually, primes have lpf[p]=p, composites have lpf[c]=largest prime.
    # If interrupted at i, then numbers that only have prime factors >= i might be wrong.
    # But for 10^6, it's fast. Let's proceed and add a verification step if needed.
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Extract the relevant slice
    vals_n = lpf[10000:1000001]
    vals_np1 = lpf[10001:1000002]
    
    # Compute max for each pair
    max_vals = np.maximum(vals_n, vals_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_vals)
    best_n = 10000 + min_idx
    
    # Verification step: Use exact trial division to verify the result
    # This ensures correctness even if the sieve was somehow incomplete
    def largest_prime_factor(x):
        """Compute the largest prime factor of x via trial division."""
        if x <= 1:
            return 1
        max_pf = 1
        # factor out 2
        while x % 2 == 0:
            max_pf = 2
            x //= 2
        # factor out odd numbers up to sqrt(x)
        f = 3
        while f * f <= x:
            while x % f == 0:
                max_pf = f
                x //= f
            f += 2
        if x > 1:
            max_pf = x
        return max_pf
    
    # Verify the best candidate
    if time.time() < deadline:
        verified_val = max(largest_prime_factor(best_n), largest_prime_factor(best_n + 1))
        sieve_val = max(lpf[best_n], lpf[best_n + 1])
        
        # If there's a discrepancy, do a more thorough search
        if abs(verified_val - sieve_val) > 1:
            # Sieve was incomplete, do exact search
            best_n_exact = 10000
            best_val_exact = float('inf')
            
            for n in range(10000, 1000001):
                if time.time() > deadline - 5:
                    break
                lp_n = largest_prime_factor(n)
                lp_np1 = largest_prime_factor(n + 1)
                val = max(lp_n, lp_np1)
                if val < best_val_exact:
                    best_val_exact = val
                    best_n_exact = n
            
            best_n = best_n_exact
    
    return int(best_n)
