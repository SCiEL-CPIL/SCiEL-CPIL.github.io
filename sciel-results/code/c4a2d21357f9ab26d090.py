import time
import numpy as np
from fractions import Fraction
from collections import defaultdict

def _calc_sigma_simple(n):
    """Calculate sigma(n) using trial division for verification."""
    if n == 1:
        return 1
    res = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            term = 1
            power = 1
            while temp % d == 0:
                temp //= d
                power *= d
                term += power
            res *= term
        d += 1
    if temp > 1:
        res *= (1 + temp)
    return res

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.85
    hard_stop = t0 + LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    N = 10000000
    fallback = [11, 6]  # sigma(11)=12, sigma(6)=12. |11/6 - 3/2| = |1.8333 - 1.5| = 0.3333

    try:
        # Strategy: Enumerate b values, compute sigma(b), then find a values
        # with same sigma that are close to 1.5*b.
        
        # First, compute sigma for all numbers up to N using a sieve
        # but only store what we need.
        
        # Compute sigma via sieve: sigma[n] = sum of divisors
        sigma_arr = np.zeros(N + 1, dtype=np.int64)
        
        # Efficient sieve: for each d, add d to all multiples of d
        # But this is O(N log N) which might be too slow for N=10^7
        # Let's use a more efficient approach with numpy
        
        # Actually, let's try a different strategy:
        # Generate candidate pairs by considering the ratio constraint directly.
        # We want a/b ≈ 3/2, so a ≈ 1.5*b
        # For each b, we need sigma(a) = sigma(b) where a is near 1.5*b
        
        # Step 1: Compute sigma for all numbers up to N
        # Use a segmented or optimized approach
        
        # Let's compute sigma using the multiplicative property
        # First, find smallest prime factor (SPF)
        spf = np.zeros(N + 1, dtype=np.int32)
        primes = []
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                spf[i * p] = p
        
        if time_left() < 30.0:
            return fallback
        
        # Now compute sigma using SPF
        sigma_arr[1] = 1
        for i in range(2, N + 1):
            p = spf[i]
            if i % (p * p) == 0:
                # p^2 divides i
                sigma_arr[i] = sigma_arr[i // p] * (p + 1)
            else:
                # p does not divide i//p
                sigma_arr[i] = sigma_arr[i // p] * (p + 1)
        
        if time_left() < 20.0:
            return fallback
        
        # Group numbers by their sigma value
        # Only keep groups with at least 2 elements
        sigma_groups = defaultdict(list)
        for n in range(2, N + 1):
            s = sigma_arr[n]
            if s <= N * 2:  # sigma values can be large, but we only care about reasonable ones
                sigma_groups[s].append(n)
        
        if time_left() < 15.0:
            return fallback
        
        best_val = float('inf')
        best_pair = None
        
        # For each group, find pairs (a, b) that minimize |a/b - 3/2|
        # |a/b - 3/2| = |2a - 3b| / (2b)
        # We want to minimize this
        
        for s_val, nums in sigma_groups.items():
            if time_left() < 5.0:
                break
            
            if len(nums) < 2:
                continue
            
            nums_sorted = sorted(nums)
            L = len(nums_sorted)
            
            # For each b, find a closest to 1.5*b
            # Use two pointers or binary search
            
            for idx_b in range(L):
                b = nums_sorted[idx_b]
                target_a = 1.5 * b
                
                # Find closest a in nums_sorted to target_a
                # Use binary search
                import bisect
                pos = bisect.bisect_left(nums_sorted, target_a)
                
                candidates = []
                if pos < L:
                    candidates.append(nums_sorted[pos])
                if pos > 0:
                    candidates.append(nums_sorted[pos - 1])
                
                for a in candidates:
                    if a == b:
                        continue
                    if 2 * a == 3 * b:
                        continue  # exact 3/2 ratio is invalid
                    
                    val = abs(2 * a - 3 * b) / (2 * b)
                    if val < best_val:
                        best_val = val
                        best_pair = (a, b)
                
                # Early termination if we've found a very good pair
                if best_val < 1e-6:
                    break
            
            if best_val < 1e-6:
                break
        
        if best_pair is None:
            return fallback
        
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
        
        # Verify the pair
        if _calc_sigma_simple(a) != _calc_sigma_simple(b):
            return fallback

        return [a, b]

    except Exception:
        return [11, 6]
