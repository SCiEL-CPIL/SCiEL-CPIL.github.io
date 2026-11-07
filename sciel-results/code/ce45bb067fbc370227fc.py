"""Search for a pair (a,b) with sigma(a)=sigma(b) minimizing |a/b - 3/2|."""
import time
import math
import sys
from fractions import Fraction

MAX_N = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    # --- Sieve for sigma(n) using linear sieve (smallest prime factor) ---
    n = MAX_N
    # We'll use arrays for SPF and sigma
    # For memory, use arrays from array module or just lists
    # SPF: smallest prime factor for each number
    # sigma: sum of divisors
    
    # Linear sieve approach
    spf = list(range(n + 1))  # smallest prime factor
    sigma = [0] * (n + 1)
    sigma[1] = 1
    
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Actually, let's use a more memory-efficient approach
    # Use array('I') or just lists
    # For n=10^7, lists of ints use ~80MB each in Python... might be too much
    # Let's use array module for memory efficiency
    
    from array import array
    
    spf = array('I', [0]) * (n + 1)
    sigma = array('Q', [0]) * (n + 1)  # unsigned long long for sigma values
    sigma[1] = 1
    
    primes = []
    is_composite = bytearray(n + 1)
    
    for i in range(2, n + 1):
        if not is_composite[i]:
            primes.append(i)
            spf[i] = i
            sigma[i] = 1 + i  # sigma of prime p is 1+p
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            is_composite[i * p] = 1
            spf[i * p] = p
            if i % p == 0:
                # p divides i, so sigma(i*p) = sigma(i) * (1 + p)
                # Wait, that's not right. sigma is multiplicative.
                # If i = p^k * m where gcd(m,p)=1, then i*p = p^(k+1)*m
                # sigma(i*p) = sigma(p^(k+1)) * sigma(m) = (1+p+...+p^(k+1)) * sigma(m)
                # sigma(i) = sigma(p^k) * sigma(m) = (1+p+...+p^k) * sigma(m)
                # So sigma(i*p) = sigma(i) * (1+p+...+p^(k+1))/(1+p+...+p^k)
                # This is complex. Let's use a different approach.
                pass
            else:
                # p does not divide i, so sigma(i*p) = sigma(i) * sigma(p) = sigma(i) * (1+p)
                sigma[i * p] = sigma[i] * (1 + p)
    
    # The above linear sieve for sigma is tricky. Let me use a simpler sieve approach.
    # Standard sieve for sigma:
    # sigma(n) = sum_{d|n} d
    # We can compute it by iterating over multiples
    
    # Let's restart with a cleaner approach
    # Use the formula: sigma(n) = product over prime powers
    # Or use the sieve: for each i, for each multiple j of i, add i to sigma[j]
    
    # This is O(n log n) which for n=10^7 is about 10^7 * ln(10^7) ≈ 1.6 * 10^8 operations
    # That might be too slow in Python. Let's try the linear sieve properly.
    
    # Actually, let me use a different strategy:
    # 1. Compute sigma using a modified sieve
    # 2. Group numbers by sigma value
    # 3. For each group, find pairs that minimize |2a - 3b|
    
    # Let's use NumPy for speed
    import numpy as np
    
    t1 = time.time()
    
    # Compute sigma using numpy
    sigma_arr = np.zeros(n + 1, dtype=np.uint64)
    for i in range(1, n + 1):
        sigma_arr[i::i] += i
    
    t2 = time.time()
    print(f"Sigma computation: {t2-t1:.2f}s", file=sys.stderr)
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time, return a default valid pair
        # Find a simple pair with equal sigma
        # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12
        # sigma(11)=12, so sigma(6)=sigma(11)=12
        return [6, 11]
    
    # Now group by sigma value
    # We need to find for each sigma value, the numbers that have that sigma
    # Then for each group, find pairs (a,b) minimizing |2a - 3b|
    
    # The key insight: for a fixed sigma value S, we have a set of numbers {x1, x2, ...}
    # We want to find xi, xj such that |2*xi - 3*xj| is minimized
    # This is equivalent to finding xi/xj closest to 3/2
    
    # Strategy: For each group, sort the numbers. Then for each element a in the group,
    # we want b such that b ≈ (2/3)*a. We can binary search for this.
    
    # But first, let's collect the groups. This could use a lot of memory.
    
    # Alternative: sort all numbers by sigma value, then process consecutive runs.
    
    # Create an array of (sigma[i], i) for i=1..n, sort by sigma
    # This is expensive in Python. Let's use numpy.
    
    indices = np.arange(1, n + 1)
    # Sort indices by sigma value
    order = np.argsort(sigma_arr[1:], kind='stable')
    sorted_sigma = sigma_arr[1:][order]
    sorted_indices = indices[order]
    
    t3 = time.time()
    print(f"Sorting: {t3-t2:.2f}s", file=sys.stderr)
    
    if time.time() > deadline - 20:
        return [6, 11]
    
    # Now iterate through the sorted array, finding groups of equal sigma
    # For each group, find the best pair
    
    best_err = float('inf')
    best_a = 6
    best_b = 11
    best_num = 1  # numerator of |a/b - 3/2| = |2a-3b|
    best_den = 2 * 6 * 11  # denominator
    
    i = 0
    m = len(sorted_sigma)
    while i < m:
        j = i + 1
        while j < m and sorted_sigma[j] == sorted_sigma[i]:
            j += 1
        # Group is sorted_indices[i:j]
        group_size = j - i
        if group_size >= 2:
            group = sorted_indices[i:j]
            # For each a in group, find b in group minimizing |2a - 3b|
            # b ≈ (2/3)a
            # Since group is sorted, we can use binary search
            for k in range(group_size):
                a = group[k]
                target_b = (2 * a) / 3
                # Binary search for target_b in group
                # group is sorted
                pos = np.searchsorted(group, target_b)
                # Check pos and pos-1
                for p in [pos - 1, pos]:
                    if 0 <= p < group_size:
                        b = group[p]
                        # Compute |2a - 3b| / (2ab)
                        diff = abs(2 * a - 3 * b)
                        if diff == 0:
                            # Perfect match, but |a/b - 3/2| = 0 is invalid
                            # We need a non-zero error
                            continue
                        err = diff / (2 * a * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            best_num = diff
                            best_den = 2 * a * b
        i = j
    
    t4 = time.time()
    print(f"Group processing: {t4-t3:.2f}s, best_err={best_err:.2e}, a={best_a}, b={best_b}", file=sys.stderr)
    
    # Verify the result
    assert best_a <= 10000000 and best_b <= 10000000
    assert isinstance(best_a, (int, np.integer))
    assert isinstance(best_b, (int, np.integer))
    
    # Convert to Python int
    best_a = int(best_a)
    best_b = int(best_b)
    
    # Double-check sigma values
    # Recompute sigma for best_a and best_b to verify
    def sigma_check(x):
        s = 0
        for d in range(1, int(math.isqrt(x)) + 1):
            if x % d == 0:
                s += d
                if d != x // d:
                    s += x // d
        return s
    
    sa = sigma_check(best_a)
    sb = sigma_check(best_b)
    if sa != sb:
        print(f"ERROR: sigma({best_a})={sa}, sigma({best_b})={sb}", file=sys.stderr)
        return [6, 11]
    
    # Check that error is not zero
    if 2 * best_a == 3 * best_b:
        print("ERROR: zero error", file=sys.stderr)
        return [6, 11]
    
    return [best_a, best_b]
