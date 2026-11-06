# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt
from bisect import bisect_left, bisect_right

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    # Sieve for square-free
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[0] = False
    is_sq_free[1] = True  # 1 is square-free
    
    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return powerful

def _count_pairs_for_N(N, powerful_list, powerful_set):
    """Count number of pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    n = len(powerful_list)
    for i in range(n):
        a = powerful_list[i]
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
    return count

def construct():
    t0 = time.time()
    deadline = t0 + 140  # 140s safety margin
    
    best_result = None
    best_count = 0
    
    # Strategy: generate powerful numbers up to a large limit,
    # then search for N with many representations using direct counting.
    # Powerful numbers are sparse: ~sqrt(limit) count.
    # For limit = 10^8, ~10^4 powerful numbers.
    # For each candidate N, counting pairs is O(sqrt(limit) * log(sqrt(limit))).
    
    # First, generate powerful numbers up to a reasonable limit
    # We'll try progressively larger limits
    limits_to_try = [50_000_000, 100_000_000, 200_000_000]
    
    for limit in limits_to_try:
        if time.time() > deadline - 60:
            break
        
        t_gen = time.time()
        powerful = _generate_powerful(limit)
        gen_time = time.time() - t_gen
        n_powerful = len(powerful)
        
        if n_powerful < 2:
            continue
        
        powerful_list = sorted(powerful)
        powerful_set = powerful
        
        # Now search for N with many representations.
        # Candidate N values: sums of two powerful numbers.
        # The number of powerful numbers is ~sqrt(limit), so there are ~limit/2 possible sums,
        # but we can't check all. Instead, we focus on N values that are likely to have many reps.
        
        # Heuristic: N around the "center" of the sum distribution tends to have more reps.
        # The sum of two powerful numbers ranges from 2 to 2*limit.
        # The density of sums is highest around the middle.
        
        # Better approach: sample candidate N values and count pairs.
        # Since powerful numbers are sparse, most N have 0 or 1 pair.
        # We need to find N with many pairs.
        
        # Strategy: For each powerful number a, consider N = a + b for various b.
        # But that's O(n^2) which is too slow for n = 10^4.
        
        # Alternative: Use the fact that if we have sorted powerful numbers,
        # we can do a two-pointer style search for specific N.
        # But we need to find the BEST N, not count for a given N.
        
        # New approach: Random sampling of N values in the range [2, 2*limit],
        # count pairs for each, and track the best.
        # With ~10^4 powerful numbers, counting for one N takes ~10^4 * log(10^4) ≈ 10^5 ops.
        # In the remaining time, we can try many N values.
        
        remaining_time = deadline - time.time() - 10  # 10s for pair generation
        if remaining_time < 1:
            continue
        
        # Estimate time per N count
        # For n = n_powerful, counting takes ~n/2 * (set lookup) ≈ n/2 * O(1)
        # Let's benchmark
        if powerful_list:
            test_N = powerful_list[0] + powerful_list[min(1, n_powerful-1)]
            t_test = time.time()
            c = _count_pairs_for_N(test_N, powerful_list, powerful_set)
            t_count = time.time() - t_test
        else:
            t_count = 0.01
            c = 0
        
        # How many N values can we try?
        max_trials = int(remaining_time / max(t_count, 0.001))
        max_trials = min(max_trials, 100000)  # Cap to avoid too many
        
        # Sample N values strategically
        # Focus on N values that are sums of "small" powerful numbers,
        # as these tend to have more representations.
        
        # Generate candidate N values as sums of pairs of small powerful numbers
        # Take the first K powerful numbers and form their sums
        K = min(n_powerful, 500)
        small_powerful = powerful_list[:K]
        
        # Generate candidate N values from sums of small powerful numbers
        candidates = set()
        for i in range(K):
            for j in range(i, K):
                s = small_powerful[i] + small_powerful[j]
                if s <= 2 * limit:
                    candidates.add(s)
        
        # Also add some random N values
        n_random = min(1000, max_trials // 10)
        for _ in range(n_random):
            r = np.random.randint(2, 2 * limit + 1)
            candidates.add(int(r))
        
        candidates = sorted(candidates)
        
        # Count pairs for each candidate, tracking the best
        best_local_N = 0
        best_local_count = 0
        best_local_pairs = []
        
        for idx, N in enumerate(candidates):
            if time.time() > deadline - 15:
                break
            
            c = _count_pairs_for_N(N, powerful_list, powerful_set)
            if c > best_local_count:
                best_local_count = c
                best_local_N = N
                # Generate pairs
                pairs = []
                for a in powerful_list:
                    if 2 * a > N:
                        break
                    b = N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_local_pairs = pairs
        
        if best_local_count > best_count:
            best_count = best_local_count
            best_result = {'N': best_local_N, 'pairs': best_local_pairs}
        
        # If we found a good result and have time, try to improve with more search
        if best_count >= 10 and time.time() < deadline - 30:
            # Try additional random N values around the best found
            extra_trials = min(5000, int((deadline - time.time() - 10) / max(t_count, 0.001)))
            for _ in range(extra_trials):
                if time.time() > deadline - 10:
                    break
                # Sample N near the best found value
                delta = int(np.random.randint(-limit // 10, limit // 10 + 1))
                N = best_local_N + delta
                if N < 2 or N > 2 * limit:
                    continue
                c = _count_pairs_for_N(N, powerful_list, powerful_set)
                if c > best_local_count:
                    best_local_count = c
                    best_local_N = N
                    pairs = []
                    for a in powerful_list:
                        if 2 * a > N:
                            break
                        b = N - a
                        if b in powerful_set:
                            pairs.append([a, b])
                    best_local_pairs = pairs
                    if best_local_count > best_count:
                        best_count = best_local_count
                        best_result = {'N': best_local_N, 'pairs': best_local_pairs}
    
    if best_result is None:
        # Fallback
        powerful = _generate_powerful(100000)
        powerful_list = sorted(powerful)
        powerful_set = powerful
        best_N = 0
        best_c = 0
        best_pairs = []
        for N in range(2, 2000):
            c = _count_pairs_for_N(N, powerful_list, powerful_set)
            if c > best_c:
                best_c = c
                best_N = N
                best_pairs = []
                for a in powerful_list:
                    if 2 * a > N:
                        break
                    b = N - a
                    if b in powerful_set:
                        best_pairs.append([a, b])
        best_result = {'N': best_N, 'pairs': best_pairs}
    
    return best_result
# EVOLVE-BLOCK-END
