import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0
    n = 210
    max_val = 10210
    
    # Precompute square-free kernels for all numbers in range
    # sf(n) is the product of primes p where v_p(n) is odd
    sf = [0] * (max_val + 1)
    
    # Sieve-like computation of square-free kernels
    # Initialize sf[i] = 1
    for i in range(2, max_val + 1):
        sf[i] = 1
        
    # For each prime p, multiply sf[i] by p for all i where v_p(i) is odd
    # We need primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, max_val + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
                
    # Efficiently compute sf using prime factorization properties
    # sf(n) = product of p where n is divisible by p but not p^2? No.
    # v_p(n) odd means n is divisible by p^(2k+1) but not p^(2k+2).
    
    # Alternative: 
    # sf(n) = n / (largest square dividing n)
    # This is easy to compute for each n if we have smallest prime factor (SPF).
    
    spf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if spf[i] == 0:
            for j in range(i, max_val + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    # Compute sf[n]
    sf_vals = [1] * (max_val + 1)
    for i in range(2, max_val + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count % 2 == 1:
                res *= p
        sf_vals[i] = res
        
    # Map kernel to list of numbers
    kernel_to_nums = defaultdict(list)
    for i in range(211, max_val + 1):
        kernel_to_nums[sf_vals[i]].append(i)
        
    # Sort lists for binary search
    for k in kernel_to_nums:
        kernel_to_nums[k].sort()
        
    best_max = max_val + 1
    best_pair = None
    
    # Helper to find square-free part of a number
    def square_free_part(x):
        # x can be large (up to 10210 * 10210 * 210 approx)
        # But we only need the square-free part.
        # We can use the spf array if x is small enough, but x might be larger than max_val.
        # However, sf(a) <= a <= 10210. 210 is small.
        # X = sf(a) * 210 <= 10210 * 210 = 2144100.
        # We can factor X using primes up to sqrt(2144100) ~ 1464.
        
        y = x
        res = 1
        p = 2
        while p * p <= y:
            count = 0
            while y % p == 0:
                y //= p
                count += 1
            if count % 2 == 1:
                res *= p
            p += 1 if p == 2 else 2 # Simple prime step, or use precomputed primes
            # Optimization: use precomputed primes
        if y > 1:
            res *= y
        return res
        
    # Precompute primes for factoring X
    primes_for_fact = []
    for i in range(2, 1500):
        if is_prime[i]:
            primes_for_fact.append(i)
            
    def square_free_part_fast(x):
        y = x
        res = 1
        for p in primes_for_fact:
            if p * p > y:
                break
            count = 0
            while y % p == 0:
                y //= p
                count += 1
            if count % 2 == 1:
                res *= p
        if y > 1:
            res *= y
        return res

    import bisect
    
    # Iterate a
    for a in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        K_a = sf_vals[a]
        X = K_a * 210
        K_target = square_free_part_fast(X)
        
        # Look up numbers with kernel K_target
        if K_target in kernel_to_nums:
            nums = kernel_to_nums[K_target]
            # Find smallest b > a
            idx = bisect.bisect_right(nums, a)
            if idx < len(nums):
                b = nums[idx]
                # Verify condition (safety check)
                prod = a * b * 210
                s = math.isqrt(prod)
                if s * s == prod and a != b:
                    curr_max = max(a, b)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_pair = (a, b)
                        
        # Early termination if we find a very good pair?
        # The minimum possible max is close to 210.
        # If best_max - 210 is small, we might stop?
        # But we want the absolute minimum.
        # However, if we find a pair with max=211, that's impossible since a,b > 210 and distinct.
        # Min distinct pair > 210 is (211, 212).
        # If we find max=212, we can stop? No, we need distinct integers.
        # If best_max is small, the chance of finding a smaller one is low.
        # But let's run until deadline or a > best_max (since if a > best_max, any b > a will have max > best_max).
        if a >= best_max:
            break

    if best_pair is None:
        # Fallback
        return [840]
        
    a, b = best_pair
    return [int(a), int(b)]
