"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict
import math

def construct():
    N = 10000000
    
    # Step 1: Compute smallest prime factor (SPF) sieve
    spf = list(range(N + 1))
    # Sieve for spf
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute sigma(n) using multiplicative property
    # For each n, decompose into p^k * m where p = spf[n]
    p_power = [1] * (N + 1)
    rest = [1] * (N + 1)
    sigma_pp = [0] * (N + 1)  # sigma of the p^k part
    sigma = [0] * (N + 1)
    sigma[1] = 1
    sigma_pp[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p * m, m divisible by p
            pp_m = p_power[m]
            p_power[n] = p * pp_m
            rest[n] = rest[m]
            sigma_pp[n] = sigma_pp[m] + p_power[n]
        else:
            # n = p * m, m not divisible by p
            p_power[n] = p
            rest[n] = m
            sigma_pp[n] = 1 + p
        sigma[n] = sigma_pp[n] * sigma[rest[n]]
    
    # Step 3: Group numbers by sigma value
    fibers = defaultdict(list)
    for n in range(1, N + 1):
        fibers[sigma[n]].append(n)
    
    # Step 4: Find best pair (a, b) with sigma[a] == sigma[b], a/b != 3/2,
    # minimizing |a/b - 3/2|
    best_error_num = None
    best_error_den = None
    best_a = None
    best_b = None
    
    for s, nums in fibers.items():
        if len(nums) < 2:
            continue
        nums.sort()
        nlen = len(nums)
        # For each b, find a close to 1.5 * b
        for i in range(nlen):
            b = nums[i]
            target = 1.5 * b
            # Binary search for target
            lo, hi = 0, nlen
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            # Check candidates around the insertion point
            for idx in (lo - 1, lo):
                if 0 <= idx < nlen:
                    a = nums[idx]
                    if a == b:
                        continue
                    # Check a/b != 3/2 exactly
                    if 2 * a == 3 * b:
                        continue
                    # Error = |a/b - 3/2| = |2a - 3b| / (2b)
                    err_num = abs(2 * a - 3 * b)
                    err_den = 2 * b
                    if err_num == 0:
                        continue
                    if best_error_num is None:
                        best_error_num = err_num
                        best_error_den = err_den
                        best_a = a
                        best_b = b
                    else:
                        # Compare err_num/err_den < best_error_num/best_error_den
                        # i.e., err_num * best_error_den < best_error_num * err_den
                        if err_num * best_error_den < best_error_num * err_den:
                            best_error_num = err_num
                            best_error_den = err_den
                            best_a = a
                            best_b = b
    
    if best_a is None:
        # Fallback: return a valid pair
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
