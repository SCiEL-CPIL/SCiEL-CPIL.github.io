"""Improved program: search for sigma-equal pairs minimizing |a/b - 3/2|."""
import time
import sys
import math
from fractions import Fraction
from collections import defaultdict

def _sigma(n):
    """Compute sum of divisors of n."""
    if n == 0:
        return 0
    total = 1
    temp = n
    f = 2
    while f * f <= temp:
        if temp % f == 0:
            p_power = 1
            while temp % f == 0:
                temp //= f
                p_power *= f
            total *= (p_power * (f + 1) - 1) // (f - 1)
        f += 1
    if temp > 1:
        total *= (temp + 1)
    return total

def _factorize(n):
    """Return list of (prime, exponent) pairs."""
    factors = []
    temp = n
    f = 2
    while f * f <= temp:
        if temp % f == 0:
            exp = 0
            while temp % f == 0:
                temp //= f
                exp += 1
            factors.append((f, exp))
        f += 1
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            pe = 1
            for _ in range(e + 1):
                new_divs.append(d * pe)
                pe *= p
        divisors = new_divs
    return divisors

def _sigma_from_factors(factors):
    """Compute sigma from prime factorization."""
    total = 1
    for p, e in factors:
        # sigma(p^e) = 1 + p + p^2 + ... + p^e = (p^(e+1) - 1)/(p - 1)
        s = 1
        pe = 1
        for _ in range(e):
            pe *= p
            s += pe
        total *= s
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s budget out of 60s
    
    # Precomputed pair from knowledge
    best_a, best_b = 122, 80
    best_err = Fraction(1, 40)  # |122/80 - 3/2| = |30.5/20 - 30/20| = 0.5/20 = 1/40
    
    MAX_N = 10000000
    
    # Strategy: For a given sigma value S, collect all n with sigma(n) = S.
    # Then for each pair (a, b) in that fiber, compute |a/b - 3/2|.
    # We want a/b close to 3/2, so a ≈ 1.5*b.
    
    # Build a dictionary: sigma_value -> list of n
    # But sigma values can be large, so we only keep small sigma values or limit size.
    
    # Actually, let's think differently. For a fixed b, we want a such that sigma(a) = sigma(b) and a ≈ 3b/2.
    # So for each b, compute sigma(b) = S, then look for a near 3b/2 with sigma(a) = S.
    
    # Alternative: iterate over possible sigma values, collect numbers, and check pairs.
    
    # Let's use a hybrid: 
    # 1. For b in a range, compute sigma(b).
    # 2. For each sigma value S, maintain a list of b's.
    # 3. For each b in the list, check if there's an a ≈ 3b/2 with sigma(a) = S.
    
    # But we need to precompute sigma for a range. Let's use a sieve-like approach for sigma.
    
    # Sieve for sigma up to MAX_N:
    # sigma[n] = sum of divisors of n
    # We can compute this using a modified sieve.
    
    print(f"Computing sigma up to {MAX_N}...", file=sys.stderr)
    sigma_arr = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma_arr[j] += i
    
    print(f"Sigma computation done at {time.time() - start_time:.1f}s", file=sys.stderr)
    
    if time.time() > deadline - 5:
        return [best_a, best_b]
    
    # Group numbers by sigma value
    # For each sigma value S, we want to find pairs (a, b) where a ≈ 1.5 * b
    # To minimize |a/b - 3/2|, we want a/b = 3/2 exactly, i.e., 2a = 3b.
    # So we look for b such that 3b/2 is an integer and sigma(3b/2) = sigma(b).
    
    # Let's check: for each b where 3b is even (i.e., b is even), a = 3b/2.
    # Check if sigma(a) == sigma(b) and a <= MAX_N.
    
    # Actually, more generally, for each b, we want a in a neighborhood of 3b/2.
    # Let's first check exact matches: a = 3b/2 when b is even.
    
    found_exact = False
    for b in range(2, min(MAX_N, 2000000) + 1, 2):  # b even
        a = 3 * b // 2
        if a <= MAX_N:
            if sigma_arr[a] == sigma_arr[b]:
                err = abs(Fraction(a, b) - Fraction(3, 2))
                if err == 0:
                    # This would be invalid per the task (raw value exactly 0 cannot be scored)
                    # So skip exact matches
                    pass
                elif err < best_err:
                    best_a, best_b = a, b
                    best_err = err
                    found_exact = True
                    print(f"Found better: a={a}, b={b}, err={best_err}", file=sys.stderr)
    
    if time.time() > deadline - 3:
        return [best_a, best_b]
    
    # Now search more broadly: for each sigma value S, collect all n with sigma(n) = S.
    # Then for each pair in the same fiber, check ratio closeness.
    
    # Build sigma -> list mapping, but limit to reasonable sigma values.
    sigma_to_nums = defaultdict(list)
    for n in range(1, min(MAX_N, 2000000) + 1):
        s = sigma_arr[n]
        if len(sigma_to_nums[s]) < 200:  # Limit list size to avoid huge fibers
            sigma_to_nums[s].append(n)
    
    # For each sigma value, check pairs
    for S, nums in sigma_to_nums.items():
        if time.time() > deadline - 1:
            break
        if len(nums) < 2:
            continue
        # Sort nums
        nums.sort()
        # For each b in nums, look for a ≈ 1.5*b
        for b in nums:
            target_a = 1.5 * b
            # Binary search or linear scan for a near target_a
            # Since nums is sorted, we can use bisect
            # But let's just check a few candidates near target_a
            lo = max(1, int(target_a * 0.8))
            hi = min(MAX_N, int(target_a * 1.2))
            # Find all a in nums within [lo, hi]
            # Use bisect
            import bisect
            idx = bisect.bisect_left(nums, lo)
            while idx < len(nums) and nums[idx] <= hi:
                a = nums[idx]
                if a != b:
                    err = abs(Fraction(a, b) - Fraction(3, 2))
                    if 0 < err < best_err:
                        best_a, best_b = a, b
                        best_err = err
                        print(f"Found better: a={a}, b={b}, err={best_err}", file=sys.stderr)
                idx += 1
    
    # Ensure types
    assert isinstance(best_a, int) and isinstance(best_b, int)
    assert 1 <= best_a <= 10000000 and 1 <= best_b <= 10000000
    assert sigma_arr[best_a] == sigma_arr[best_b] if best_a <= MAX_N and best_b <= MAX_N else True
    err = abs(Fraction(best_a, best_b) - Fraction(3, 2))
    assert err != 0, "Error cannot be exactly 0"
    
    return [best_a, best_b]
