"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using square-free sieve."""
    # A powerful number is of the form a^2 * b^3 where b is square-free
    # We need b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1.0 / 3.0)) + 1
    
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    is_sq_free[0] = False  # 0 is not square-free in our context
    
    # Generate powerful numbers: a^2 * b^3 <= limit
    # For each square-free b, a^2 <= limit / b^3
    powerful_set = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful_set.add(val)
    
    return sorted(powerful_set)

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Generate powerful numbers up to a large limit
    # The key insight: more powerful numbers -> more possible sums -> higher collision counts
    # With limit ~5e6, we get thousands of powerful numbers
    limit = 5_000_000
    powerful = _generate_powerful(limit)
    members = set(powerful)
    n_pow = len(powerful)
    
    # Count collisions: for each pair (a, b) with a <= b and a+b = N, increment count[N]
    # Use a dict to track counts
    from collections import defaultdict
    counts = defaultdict(int)
    
    # Iterate over all pairs (a, b) with a <= b
    # To avoid O(n^2), iterate a in powerful, b in powerful where a <= b
    # But we can be smarter: for each a, b must be in powerful and b >= a and a+b <= 2*limit
    # Actually a+b can be up to 2*limit, but we just count collisions
    
    # Efficient approach: for each a in powerful (sorted), for each b in powerful where b >= a:
    #   s = a + b
    #   counts[s] += 1
    # This is O(n^2) which for n ~ 20000 is ~2e8 - too slow in Python
    
    # Better: use the fact that powerful numbers are sparse.
    # For n ~ 5000 powerful numbers, n^2/2 ~ 1.25e7 pairs - feasible
    # Let's check how many we have
    if n_pow > 10000:
        # Too many for O(n^2) in Python, need to be smarter
        # Use numpy for vectorized sum counting
        import numpy as np
        arr = np.array(powerful, dtype=np.int64)
        # Compute all pair sums a[i] + a[j] for i <= j
        # This is memory intensive for large n
        # Alternative: iterate and count
        # For n=10000, n^2/2 = 5e7 - might be borderline
        # Let's use a hybrid: process in chunks
        half = n_pow // 2
        for i in range(n_pow):
            a = powerful[i]
            # b must be >= a, so j >= i
            # b in powerful[i:], sum = a + b
            # We can use numpy to compute sums and bincount
            b_arr = arr[i:]  # b >= a
            sums = a + b_arr
            # Count occurrences
            # Use np.bincount if sums are not too large
            max_sum = sums.max()
            if max_sum <= 20_000_000:
                c = np.bincount(sums, minlength=max_sum + 1)
                for s, cnt in enumerate(c):
                    if cnt > 0:
                        counts[s] += int(cnt)
            else:
                # Fallback: iterate
                for b in powerful[i:]:
                    counts[a + b] += 1
            if time.time() > deadline - 10:
                break
    else:
        for i in range(n_pow):
            a = powerful[i]
            for j in range(i, n_pow):
                b = powerful[j]
                counts[a + b] += 1
            if time.time() > deadline - 10:
                break
    
    # Find the N with maximum count
    if not counts:
        # Fallback: return a simple valid result
        N = 2
        return {'N': N, 'pairs': [[1, 1]]}
    
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Collect all pairs for best_N
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in members and a <= b:
            pairs.append([a, b])
    
    # Ensure pairs are distinct and sorted
    pairs = sorted(set(tuple(p) for p in pairs))
    pairs = [list(p) for p in pairs]
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
