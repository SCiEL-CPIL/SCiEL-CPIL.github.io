"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MAX_N = 10000000
    
    # 1. Compute sigma(n) for all n up to MAX_N using a sieve method
    # sigma(n) = sum of divisors. 
    # We can compute this by iterating d from 1 to MAX_N and adding d to all multiples of d.
    # This is O(N log N), which for 10^7 is roughly 10^7 * 16 = 1.6 * 10^8 operations.
    # In Python, this might be too slow if done naively.
    # Let's try a faster approach or check if we can optimize.
    # Actually, 10^7 is large for a pure Python sieve in 55s. 
    # A pure Python loop for d in range(1, 10^7+1) and inner loop for m in range(d, 10^7+1, d) is likely too slow.
    
    # Alternative: Use numpy if available? The prompt says NumPy is available.
    # Let's try to use NumPy to accelerate the sieve.
    
    try:
        import numpy as np
        # Initialize sigma array with zeros
        # Note: sigma(n) can be large. For n=10^7, max sigma is roughly n * log log n ~ 10^7 * 3 = 3*10^7.
        # int64 is sufficient.
        sigma_arr = np.zeros(MAX_N + 1, dtype=np.int64)
        
        # Sieve: for d from 1 to MAX_N, add d to sigma_arr[d::d]
        # This is a standard trick.
        # However, doing this in a Python loop for d is still O(N log N) Python overhead if not vectorized per d.
        # np.add.at(sigma_arr, np.arange(d, MAX_N+1, d), d) is slow due to Python loop overhead.
        
        # Let's estimate time. 10^7 iterations of the outer loop. Too slow.
        
        # We need a different strategy for the sieve or a different search method.
        # Maybe we don't need ALL sigmas.
        
        # Let's go back to the ratio constraint: 2a - 3b = k.
        # The current program checks sigma(a) == sigma(b) on the fly.
        # The bottleneck is the number of sigma calls.
        
        # New Strategy:
        # Precompute sigma for a *subset* of numbers? No, we need to find the global optimum.
        
        # Let's optimize the on-the-fly sigma calculation.
        # The current `sigma(n)` uses trial division up to sqrt(n).
        # For n=10^7, sqrt(n)=3162.
        # If we check 10,000 candidates, that's 10,000 * 3162 ~ 3*10^7 operations. This is fast.
        # Why was the current program slow or low score?
        # The current program iterates k from 1 to 1000.
        # For each k, it checks the top 1000 b values.
        # Total candidates: 1000 * 1000 = 1,000,000 candidates.
        # Each candidate requires sigma(a) and sigma(b).
        # 2,000,000 sigma calls.
        # If each sigma call takes ~1 microsecond (optimistic for Python), that's 2 seconds.
        # If it takes ~100 microseconds, that's 200 seconds. Timeout.
        
        # So the bottleneck IS the sigma calculation for many candidates.
        
        # Optimization:
        # 1. Precompute primes up to sqrt(10^7) = 3162.
        # 2. Use a faster sigma function using these primes.
        # 3. Reduce the number of candidates checked.
        
        # Even better:
        # Can we precompute sigma for all numbers in a more efficient way?
        # Using NumPy, we can do:
        # sigma_arr = np.zeros(MAX_N + 1, dtype=np.int64)
        # for i in range(1, MAX_N + 1):
        #     sigma_arr[i::i] += i
        # This loop runs 10^7 times. In Python, this is too slow.
        
        # However, we can vectorize the inner part? No, the step size changes.
        
        # Let's stick to on-the-fly sigma but make it MUCH faster.
        # Precompute smallest prime factor (SPF) up to 10^7?
        # SPF array of size 10^7 takes 80MB.
        # Computing SPF via sieve is O(N log log N).
        # In Python, a sieve for 10^7 might take 5-10 seconds if optimized.
        # Then sigma(n) can be computed in O(log n) using the SPF chain.
        
        # Let's try computing SPF up to 10^7.
        
        # Check time for SPF sieve.
        spf = np.arange(MAX_N + 1, dtype=np.int32)
        spf[0] = 0
        spf[1] = 1
        
        # Sieve of Eratosthenes variant for SPF
        for i in range(2, int(math.isqrt(MAX_N)) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i*i, MAX_N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
                        
        # This might be too slow in pure Python/NumPy mix without numba.
        # Let's check the time.
        if time.time() > deadline - 10:
             # Fallback to trial division if SPF fails
             pass
        else:
             # Use SPF to compute sigma quickly
             def sigma_spf(n):
                 if n <= 0: return 0
                 res = 1
                 while n > 1:
                     p = spf[n]
                     count = 0
                     power_sum = 1
                     current_power = 1
                     while n % p == 0:
                         n //= p
                         count += 1
                         current_power *= p
                         power_sum += current_power
                     res *= power_sum
                 return res
             
             # Now we can check many more candidates.
             
             best_a, best_b = 11, 6
             best_err = abs(11/6 - 1.5)
             
             # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b).
             # Let k = 2a - 3b.
             # We iterate k from 1 upwards.
             # For a fixed k, we want to maximize b to minimize error.
             # Constraints:
             # 1. a, b <= 10^7
             # 2. a = (3b + k) / 2 => 3b + k must be even.
             # 3. a >= 1, b >= 1.
             
             max_k_to_check = 10000 # We can check more now if sigma is fast
             
             for k in range(1, max_k_to_check + 1):
                 if time.time() > deadline:
                     break
                     
                 # Find max b
                 # 3b + k <= 20000000
                 max_b_possible = (20000000 - k) // 3
                 if max_b_possible < 1:
                     continue
                 
                 # Parity: b % 2 == k % 2
                 if max_b_possible % 2 != k % 2:
                     b_start = max_b_possible - 1
                 else:
                     b_start = max_b_possible
                     
                 # Check candidates from largest b downwards
                 # How many? With fast sigma, we can check more.
                 # But error decreases as b increases.
                 # If we find a match at b_start, it's the best for this k.
                 # If not, we check b_start - 2, etc.
                 
                 # Let's check a limited number to avoid timeout, but more than before.
                 # 1000 candidates per k * 10000 k = 10^7 sigma calls.
                 # If sigma is O(log n) ~ 10-20 steps, that's 10^8 ops. Might be tight.
                 # Let's reduce max_k or candidates per k.
                 
                 candidates_per_k = 200
                 
                 for i in range(candidates_per_k):
                     b = b_start - 2*i
                     if b < 1:
                         break
                     
                     a = (3 * b + k) // 2
                     
                     if a > 10000000 or a < 1:
                         continue
                     
                     sa = sigma_spf(a)
                     sb = sigma_spf(b)
                     
                     if sa == sb:
                         err = abs(a / b - 1.5)
                         if err < best_err:
                             best_a, best_b = a, b
                             best_err = err
                             # Found a good one for this k.
                             # Can we break?
                             # Yes, for this k, larger b is better. We started from max b.
                             # So this is the best for this k.
                             break
                             
             return [int(best_a), int(best_b)]
             
    except Exception:
        # Fallback to original slower method if NumPy/SPF fails
        def sigma(n):
            if n <= 0: return 0
            res = 1
            i = 2
            while i * i <= n:
                if n % i == 0:
                    count = 0
                    while n % i == 0:
                        n //= i
                        count += 1
                    term = (i ** (count + 1) - 1) // (i - 1)
                    res *= term
                i += 1
            if n > 1:
                res *= (n + 1)
            return res
            
        best_a, best_b = 11, 6
        best_err = abs(11/6 - 1.5)
        
        max_k_to_check = 1000
        for k in range(1, max_k_to_check + 1):
            if time.time() > deadline:
                break
            max_b_possible = (20000000 - k) // 3
            if max_b_possible < 1: continue
            if max_b_possible % 2 != k % 2:
                b_start = max_b_possible - 1
            else:
                b_start = max_b_possible
                
            for b in range(b_start, max(0, b_start - 1000), -2):
                if time.time() > deadline: break
                a = (3 * b + k) // 2
                if a < 1 or a > 10000000: continue
                if sigma(a) == sigma(b):
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_a, best_b = a, b
                        best_err = err
                        break
        return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
