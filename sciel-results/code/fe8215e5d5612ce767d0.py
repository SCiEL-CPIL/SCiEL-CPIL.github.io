"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _is_powerful(n):
    """Check if n is powerful (every prime factor appears with exponent >= 2)."""
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp == 1:
                return False
        d += 1 if d == 2 else 2
    if temp > 1:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    # Powerful numbers are of the form a^2 * b^3 where b is square-free
    # Generate by iterating over b (square-free) and a
    # b^3 <= limit => b <= limit^(1/3)
    max_b = int(round(limit ** (1/3))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # Check if b is square-free
        temp_b = b
        d = 2
        sq_free = True
        while d * d <= temp_b:
            if temp_b % (d * d) == 0:
                sq_free = False
                break
            d += 1 if d == 2 else 2
        if not sq_free:
            continue
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers up to a large limit
    # Start with a reasonable limit and increase if time permits
    limit = 50_000_000  # 5 * 10^7
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    # Phase 2: Count representations for all sums using a bucketing approach
    # For each powerful number a, and each powerful number b, a+b is a sum
    # We want to find N with maximum count of pairs (a,b) where a <= b and a+b=N
    
    # Strategy: Use the sorted list and for each a, find how many b satisfy a <= b and b = N - a
    # More efficiently: count all pairs (a,b) with a <= b, group by sum
    
    # For K ~ 15000, K^2/2 ~ 112M pairs which is too slow in Python
    # Use a smarter approach: for each a in powerful, the valid b range is [a, limit-a]
    # Count how many powerful numbers fall in that range for each sum
    
    # Alternative: iterate over sums in a smart way
    # For each a (small), b ranges from a to limit-a
    # Sum = a + b, so for fixed a, as b varies over powerful numbers in [a, limit-a],
    # we get sums a + b
    
    # Let's use a histogram approach with numpy for speed
    import numpy as np
    
    arr = np.array(powerful, dtype=np.int64)
    K = len(arr)
    
    # For each a[i], valid b[j] must satisfy b[j] >= a[i] and b[j] <= limit - a[i]
    # Sum = a[i] + b[j]
    
    # We'll process in chunks to manage memory
    # For each i, find j range where arr[j] >= arr[i] and arr[j] <= limit - arr[i]
    
    # Use binary search to find valid j ranges
    from bisect import bisect_left, bisect_right
    
    sum_counts = defaultdict(int)
    
    # Process each a
    for i in range(K):
        a = arr[i]
        if a > limit // 2:
            break
        # b must be >= a and <= limit - a
        j_start = bisect_left(arr, a)
        j_end = bisect_right(arr, limit - a)
        if j_start >= j_end:
            continue
        # For each b in arr[j_start:j_end], sum = a + b
        # This is still O(K^2) in worst case
        # But with K~15000, and average range maybe ~K/2, this is ~75M iterations
        # Too slow in pure Python
        
        # Instead, let's use numpy to compute all sums for this a at once
        b_vals = arr[j_start:j_end]
        sums = a + b_vals
        # Count each sum
        # Use np.unique with return_counts
        unique_sums, counts = np.unique(sums, return_counts=True)
        for s, c in zip(unique_sums, counts):
            sum_counts[int(s)] += int(c)
        
        if time.time() > deadline - 30:
            break
    
    # Find the best N
    if not sum_counts:
        # Fallback: return a simple result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_counts.keys(), key=lambda n: sum_counts[n])
    best_count = sum_counts[best_N]
    
    # Generate the actual pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    # Verify
    assert len(pairs) == best_count, f"Pair count mismatch: {len(pairs)} vs {best_count}"
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
