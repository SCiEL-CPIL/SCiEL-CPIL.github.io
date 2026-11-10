"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction

MAX_N = 10_000_000
TARGET = Fraction(3, 2)
DEADLINE_SECONDS = 85  # safety margin before 90s kill

def construct():
    t0 = time.time()
    
    # Step 1: Compute sigma(n) for all n up to MAX_N using sieve method
    # sigma(n) = sum of all positive divisors of n
    # Use the multiplicative property with smallest prime factor sieve
    
    N = MAX_N
    sigma = [0] * (N + 1)
    sigma[0] = 0
    sigma[1] = 1
    
    # Sieve: for each prime p, multiply sigma by (p^(k+1)-1)/(p-1) for p^k || n
    # More efficient: initialize sigma[n] = 1, then for each i, add i to multiples
    # But that's O(N log N) which is too slow for N=10M in 85s
    
    # Better approach: use the formula sigma(n) = product over prime powers
    # Compute smallest prime factor (SPF) first, then compute sigma multiplicatively
    
    spf = list(range(N + 1))  # smallest prime factor
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute sigma(n) for all n using SPF
    # sigma(n) = sigma(n / p^k) * (1 + p + p^2 + ... + p^k) where p^k || n
    sigma_arr = [1] * (N + 1)
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        # Count how many times p divides n
        if m % p == 0:
            # p divides m, so n has at least p^2
            # sigma(n) = sigma(m) * (1 + p) if m is not divisible by p... 
            # Actually need to be more careful
            pass
    
    # Let me use a different approach: direct sieve for sigma
    # sigma[n] starts at 1 (the divisor 1)
    # For each i from 2 to N, add i to sigma[j] for all j that are multiples of i
    # This is O(N * H(N)) ≈ O(N log N) ≈ 10M * ~16 = 160M operations - too slow
    
    # Alternative: use the fact that we only need to find pairs with equal sigma
    # and ratio close to 3/2. Let's be smarter.
    
    # Strategy: For b in some range, we want a ≈ 3b/2 with sigma(a) = sigma(b)
    # So iterate over possible sigma values and find pairs
    
    # Most efficient: compute sigma for all n, group by sigma value
    # For each group, find pair (a,b) with a > b and a/b closest to 3/2
    
    # Let's use a segmented or optimized approach
    # Actually, for N=10M, the O(N log N) sieve might work in ~30-60s in C but in Python it's too slow.
    
    # Let me try a different structural approach:
    # Generate candidate pairs directly.
    # For ratio r = a/b ≈ 3/2, a = round(3b/2) or nearby values.
    # For each b, check a = floor(3b/2), ceil(3b/2), and a few neighbors.
    # Compute sigma(a) and sigma(b) and check equality.
    
    # This is O(N * K) where K is the number of neighbors to check per b.
    # With K ~ 20 and N = 10M, that's 200M sigma computations - too slow.
    
    # Better: precompute sigma for all n up to N, then do the search.
    # Let's optimize the sigma computation.
    
    # Use array module for memory efficiency
    import array
    
    # Sieve for sigma: O(N log N) but with small constant
    # sigma[n] = sum of divisors
    # Initialize sigma = [0]*(N+1), then for i in 1..N: for j in i, 2i, 3i, ...: sigma[j] += i
    # This is sum over i of N/i = N * H(N) ≈ 10M * 16.7 ≈ 167M additions
    
    # In Python this will be too slow. Let me think of a better way.
    
    # Use NumPy for vectorized operations
    import numpy as np
    
    # Compute sigma using NumPy
    # For each i, add i to all multiples of i
    # sigma = np.zeros(N+1, dtype=np.int64)
    # for i in range(1, N+1):
    #     sigma[i::i] += i
    # This loop is still O(N log N) but NumPy makes each inner step fast.
    # For i=1: 10M ops, i=2: 5M ops, ..., total ~167M ops
    # NumPy can do ~100M simple ops per second, so ~2 seconds? Maybe 5-10s.
    
    # Let's try it with a deadline check
    if time.time() - t0 > DEADLINE_SECONDS:
        return [11, 6]
    
    sigma_np = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() - t0 > DEADLINE_SECONDS - 10:
            break
        sigma_np[i::i] += i
    
    if time.time() - t0 > DEADLINE_SECONDS - 5:
        return [11, 6]
    
    # Now find the best pair
    # For each b, we want a ≈ 3b/2 with sigma_np[a] == sigma_np[b]
    # Iterate b from 1 to N, check a in a small window around 3b/2
    
    best_err = None
    best_pair = (11, 6)
    
    # Check the seed first
    if sigma_np[11] == sigma_np[6]:
        err = abs(Fraction(11, 6) - TARGET)
        best_err = err
        best_pair = (11, 6)
    
    # Search: for b in range, check a near 3b/2
    # To be efficient, iterate b and check a few candidates
    step = 1
    # Check all b, but with a limited window for a
    window = 50  # check a in [3b/2 - window, 3b/2 + window]
    
    # This is still 10M * 100 = 1B operations, too slow
    # Instead: group by sigma value
    
    # Create a dictionary mapping sigma value to list of n's
    # But 10M entries in a dict is memory-heavy
    
    # Alternative: for each b, compute target_a = 3b//2, check sigma_np[target_a] == sigma_np[b]
    # and a few neighbors. This is O(N * 2*window) sigma lookups.
    # With window=10, that's 20 * 10M = 200M lookups. In NumPy this might be feasible.
    
    # Vectorized approach:
    # For each b, a_target = 3*b//2
    # Check sigma_np[a_target] == sigma_np[b]
    
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    a_target = (3 * b_arr) // 2
    
    # Filter: a_target must be in range [1, N] and a_target != b (to avoid a=b which gives ratio 1)
    # Also a_target > 0
    valid = (a_target >= 1) & (a_target <= N) & (a_target != b_arr)
    
    a_target_valid = a_target[valid]
    b_valid = b_arr[valid]
    
    # Check sigma equality
    sigma_a = sigma_np[a_target_valid]
    sigma_b = sigma_np[b_valid]
    matches = sigma_a == sigma_b
    
    if np.any(matches):
        a_matches = a_target_valid[matches]
        b_matches = b_valid[matches]
        
        # Compute error |a/b - 3/2| for each match
        # err = |2a - 3b| / (2b)
        # Minimize |2a - 3b| / b
        # = |2a - 3b| / b
        
        num = np.abs(2 * a_matches - 3 * b_matches)
        den = 2 * b_matches
        # We want to minimize num/den
        # Sort by num/den
        # To avoid floating point issues, compare fractions
        # For large arrays, use a proxy: sort by num/den
        
        # Compute ratio as float for sorting
        ratios = num.astype(np.float64) / den.astype(np.float64)
        min_idx = np.argmin(ratios)
        
        a_best = int(a_matches[min_idx])
        b_best = int(b_matches[min_idx])
        
        if a_best != b_best and a_best <= N and b_best <= N:
            err = abs(Fraction(a_best, b_best) - TARGET)
            if best_err is None or err < best_err:
                best_err = err
                best_pair = (a_best, b_best)
    
    # Also check a few neighbors of 3b/2 for better results
    # For the best b found, check neighbors
    if best_pair[1] > 0:
        b_center = best_pair[1]
        a_center = 3 * b_center // 2
        for da in range(-20, 21):
            a_cand = a_center + da
            if 1 <= a_cand <= N and a_cand != b_center:
                if sigma_np[a_cand] == sigma_np[b_center]:
                    err = abs(Fraction(a_cand, b_center) - TARGET)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a_cand, b_center)
    
    # Verify the result
    a_final, b_final = best_pair
    if sigma_np[a_final] != sigma_np[b_final]:
        # Fallback to seed
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - TARGET)
    if err_final == 0:
        # Invalid, fallback
        return [11, 6]
    
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
