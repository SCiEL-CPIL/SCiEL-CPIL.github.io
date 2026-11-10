from itertools import product
from math import gcd
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # LCM of 1..30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    # L = 2329089562800
    
    # Precompute weights: w[i] = L // (i+1) for i=0..29
    w = [L // (i + 1) for i in range(30)]
    
    # Meet in the middle: first 15 terms, last 15 terms
    half1_weights = w[:15]
    half2_weights = w[15:]
    
    # Generate all possible sums for half 1
    # Each coefficient in {-1, 0, 1}
    # Sum = sum(c[i] * half1_weights[i])
    
    # Use iterative generation
    sums1 = [0]
    for weight in half1_weights:
        new_sums = []
        for s in sums1:
            new_sums.append(s)
            new_sums.append(s + weight)
            new_sums.append(s - weight)
        sums1 = new_sums
    
    # sums1 has 3^15 = 14348907 elements
    # Sort and use for lookup
    sums1.sort()
    
    # For efficiency, use a set for O(1) lookup
    sums1_set = set(sums1)
    
    # Now for each possible sum in half 2, check if we can get total = 1 or -1
    # Total = s1 + s2, want |total| = 1
    # So s1 = 1 - s2 or s1 = -1 - s2
    
    best_abs = None
    best_coeffs = None
    
    # Generate half 2 sums and check
    sums2 = [0]
    for weight in half2_weights:
        new_sums = []
        for s in sums2:
            new_sums.append(s)
            new_sums.append(s + weight)
            new_sums.append(s - weight)
        sums2 = new_sums
    
    # Check for |total| = 1
    found = False
    for s2 in sums2:
        for target in [1, -1]:
            needed = target - s2
            if needed in sums1_set:
                found = True
                break
        if found:
            break
        if time.time() > deadline - 5:
            break
    
    if found:
        # Need to recover the actual coefficients
        # This is tricky - let me use a different approach
        pass
    
    # Alternative: if we can't easily recover coefficients, use a different strategy
    # Let me try a direct search with pruning
    
    # Actually, let me try a simpler approach: random search + local optimization
    import random
    random.seed(42)
    
    best_val = None
    best_c = None
    
    # Initial random attempts
    for _ in range(100000):
        if time.time() > deadline - 10:
            break
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        # Compute sum as fraction
        num = 0
        den = 1
        for i in range(30):
            if c[i] != 0:
                # Add c[i]/(i+1)
                num = num * (i + 1) + c[i] * den
                den = den * (i + 1)
                g = gcd(abs(num), den)
                if g > 1:
                    num //= g
                    den //= g
        if num != 0:
            abs_val = abs(num) / den
            if best_val is None or abs_val < best_val:
                best_val = abs_val
                best_c = c[:]
    
    # Local optimization: flip one coefficient at a time
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(30):
            old_val = best_c[i]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                best_c[i] = new_val
                # Compute sum
                num = 0
                den = 1
                for j in range(30):
                    if best_c[j] != 0:
                        num = num * (j + 1) + best_c[j] * den
                        den = den * (j + 1)
                        g = gcd(abs(num), den)
                        if g > 1:
                            num //= g
                            den //= g
                if num != 0:
                    abs_val = abs(num) / den
                    if best_val is None or abs_val < best_val:
                        best_val = abs_val
                        improved = True
                else:
                    # Zero sum, invalid
                    best_c[i] = old_val
                if time.time() > deadline - 5:
                    break
            best_c[i] = old_val
            if time.time() > deadline - 5:
                break
    
    if best_c is None or best_val is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    return best_c
