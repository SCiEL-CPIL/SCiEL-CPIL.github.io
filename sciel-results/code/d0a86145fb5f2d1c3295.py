import time
import numpy as np
from math import isqrt
from bisect import bisect_left, bisect_right

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    Powerful = a^2 * b^3 where b is square-free.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 2
    # Sieve for square-free numbers up to max_b
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

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

def construct():
    t0 = time.time()
    deadline = t0 + 115  # 115s safety margin (evaluator kills at 150s)

    best_result = None
    best_count = 0

    # Strategy: Generate powerful numbers up to a large limit, then use
    # sorted-list + binary search to find N with many representations.
    # This avoids the memory constraint of FFT and allows much larger limits.
    
    # For limit L, we can search N in [2, 2L].
    # Number of powerful numbers up to L is roughly O(L^(1/3) * sqrt(L)) ~ O(L^(5/6))
    # For L = 50M: ~50M^(5/6) ≈ 50M^0.833 ≈ very large, but actual count is smaller.
    # Let's estimate: powerful numbers up to 10^7 ≈ 1000, up to 10^8 ≈ 10000, 
    # up to 10^9 ≈ 100000. So for L=50M, we might have ~20000-50000 powerful numbers.
    
    # Try progressively larger limits
    limits_to_try = [20_000_000, 50_000_000, 100_000_000]
    
    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break

        # Generate powerful numbers
        powerful = _generate_powerful(limit)
        n_powerful = len(powerful)
        
        if n_powerful < 10:
            continue

        powerful_list = sorted(powerful)
        powerful_set = powerful

        # For each powerful number a (as the smaller element), find how many b >= a
        # such that a + b = N and b is powerful.
        # Instead of iterating all N, we can use a different approach:
        # For each pair (a, b) with a <= b, a+b = N. Count occurrences of each N.
        # But that's O(n^2) which is too slow for n=50000.
        
        # Better approach: For each a in powerful_list, for each b in powerful_list with b >= a,
        # compute N = a+b and count. But this is still O(n^2).
        
        # Alternative: Use the fact that we want to find N with max representations.
        # We can sample or use a heuristic.
        
        # Actually, let's use a smarter approach:
        # For each a in powerful_list, the possible N values are a + b for b in powerful_list with b >= a.
        # We can use a hash map to count N values, but we need to be efficient.
        
        # For n=50000, n^2 = 2.5e9 which is too slow.
        # We need a better method.
        
        # Let's use a different strategy: 
        # 1. Generate powerful numbers up to limit
        # 2. For a subset of candidate N values (e.g., multiples of small numbers, or 
        #    values near the "center" of the range), count representations efficiently.
        
        # Actually, the most efficient approach for large n is:
        # - Sort powerful_list
        # - For each a in powerful_list, use binary search to find b values such that a+b = N
        # - But we need to do this for many N values.
        
        # Let's try a hybrid: use FFT for smaller limits, and for larger limits,
        # use a sampling-based approach.
        
        # For now, let's just try to find the best N by checking a range of N values
        # around where we expect many representations.
        
        # The number of representations of N as sum of two powerful numbers is roughly
        # proportional to the "density" of powerful numbers near N/2.
        
        # Let's check N values in a range and count representations for each.
        # For each N, count = number of a in powerful such that a <= N/2 and N-a in powerful.
        
        # To make this efficient, we'll check a limited number of N values.
        
        # Estimate: powerful numbers up to L have density ~ c * L^(-1/3) or similar.
        # The number of representations of N is roughly (number of powerful <= N/2) * (density near N).
        
        # Let's just try N values from 2 to 2*limit and pick the best, but only if feasible.
        # For limit=20M, 2*limit=40M, which is too many to check individually.
        
        # Instead, let's use a different approach:
        # For each a in powerful_list, add a to all b in powerful_list where b >= a.
        # Use numpy for vectorized operations.
        
        if n_powerful < 100000:  # Should be true for our limits
            # Use numpy to compute all sums efficiently
            # But n^2 might be too large. For n=50000, n^2=2.5e9, too large.
            # For n=5000, n^2=25e6, feasible.
            # For n=10000, n^2=100e6, feasible with numpy.
            
            if n_powerful <= 15000:
                # Compute all pairwise sums using numpy
                arr = np.array(powerful_list, dtype=np.int64)
                # We need a <= b, so for each i, j >= i
                # Create upper triangular indices
                i_idx, j_idx = np.triu_indices(n_powerful)
                sums = arr[i_idx] + arr[j_idx]
                # Count occurrences
                # Use np.bincount for counting
                max_sum = int(sums.max())
                if max_sum <= 10**18 and max_sum < 200_000_000:  # Reasonable size
                    counts = np.bincount(sums)
                    max_N = int(np.argmax(counts))
                    max_count = int(counts[max_N])
                    
                    if max_count > best_count:
                        best_count = max_count
                        # Generate the actual pairs
                        pairs = []
                        for a in powerful_list:
                            if 2 * a > max_N:
                                break
                            b = max_N - a
                            if b in powerful_set:
                                pairs.append([a, b])
                        best_result = {'N': max_N, 'pairs': pairs}
                else:
                    # max_sum too large for bincount, use a different approach
                    pass
            else:
                # n_powerful > 15000, use a sampling approach
                # Check a limited number of N values
                # Sample N values from the range [2, 2*limit]
                # Focus on N values where we expect many representations
                
                # For each sampled N, count representations using binary search
                n_samples = 10000
                n_min = 2
                n_max = 2 * limit
                # Sample N values, biased toward the center where density is higher
                n_candidates = np.random.randint(n_min, n_max, size=n_samples)
                # Also include some specific candidates
                n_candidates = np.unique(np.concatenate([n_candidates, 
                                                         [n_min + i * (n_max - n_min) // 100 for i in range(101)]]))
                
                for N_candidate in n_candidates:
                    if time.time() > deadline - 15:
                        break
                    N_candidate = int(N_candidate)
                    # Count a in powerful such that a <= N_candidate/2 and N_candidate - a in powerful
                    count = 0
                    for a in powerful_list:
                        if a > N_candidate // 2:
                            break
                        b = N_candidate - a
                        if b in powerful_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        pairs = []
                        for a in powerful_list:
                            if 2 * a > N_candidate:
                                break
                            b = N_candidate - a
                            if b in powerful_set:
                                pairs.append([a, b])
                        best_result = {'N': N_candidate, 'pairs': pairs}

        if time.time() > deadline - 10:
            break

    if best_result is None:
        # Fallback: very small search
        limit = 100_000
        powerful = _generate_powerful(limit)
        powerful_list = sorted(powerful)
        powerful_set = powerful
        n_powerful = len(powerful_list)
        if n_powerful <= 15000:
            arr = np.array(powerful_list, dtype=np.int64)
            i_idx, j_idx = np.triu_indices(n_powerful)
            sums = arr[i_idx] + arr[j_idx]
            max_sum = int(sums.max())
            if max_sum < 200_000_000:
                counts = np.bincount(sums)
                max_N = int(np.argmax(counts))
                pairs = []
                for a in powerful_list:
                    if 2 * a > max_N:
                        break
                    b = max_N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': max_N, 'pairs': pairs}
            else:
                best_result = {'N': 2, 'pairs': [[1, 1]]}
        else:
            best_result = {'N': 2, 'pairs': [[1, 1]]}

    return best_result
