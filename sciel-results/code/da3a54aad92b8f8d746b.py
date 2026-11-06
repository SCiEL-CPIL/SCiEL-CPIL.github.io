import time
import numpy as np
from math import isqrt
from bisect import bisect_right

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

def _count_representations(N, powerful_list, powerful_set):
    """Count the number of pairs (a, b) with a <= b, a + b = N, both powerful."""
    count = 0
    for a in powerful_list:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set:
            count += 1
    return count

def construct():
    t0 = time.time()
    deadline = t0 + 115  # 115s safety margin (evaluator kills at 150s)

    best_result = None
    best_count = 0

    # Structural change: use meet-in-the-middle with large powerful number set
    # Generate powerful numbers up to a large limit, then count representations
    # for candidate N values using hash-set lookups.
    
    # The key: with a set of ~10^5 powerful numbers, we can test many candidate N
    # values efficiently. Each N test is O(n) where n is the number of powerful
    # numbers <= N/2, but with early termination.
    
    # Generate powerful numbers up to a large limit
    limit = 200_000_000  # 2 * 10^8
    powerful = _generate_powerful(limit)
    powerful_list = sorted(powerful)
    powerful_set = powerful
    n_powerful = len(powerful_list)
    
    if time.time() > deadline - 30:
        # Fallback to small search
        limit_small = 5_000_000
        powerful_small = _generate_powerful(limit_small)
        powerful_list_small = sorted(powerful_small)
        powerful_set_small = powerful_small
        # Use FFT for small case
        arr = np.zeros(limit_small + 1, dtype=np.float64)
        for p in powerful_small:
            arr[p] = 1.0
        fft_size = 1
        while fft_size < 2 * (limit_small + 1):
            fft_size *= 2
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real
        max_idx = int(np.argmax(conv[2:])) + 2
        N_best = max_idx
        conv_val = int(round(conv[N_best]))
        if N_best % 2 == 0 and N_best // 2 <= limit_small and arr[N_best // 2] > 0:
            n_pairs = (conv_val + 1) // 2
        else:
            n_pairs = conv_val // 2
        pairs = []
        for a in powerful_list_small:
            if 2 * a > N_best:
                break
            b = N_best - a
            if b in powerful_set_small:
                pairs.append([a, b])
        return {'N': N_best, 'pairs': pairs}
    
    # Strategy: find candidate N values that likely have many representations.
    # Approach: for each powerful number a, N = a + b where b is also powerful.
    # We sample sums and use a frequency dictionary to find dense clusters.
    # But O(n^2) is too slow. Instead, we use a smarter approach:
    
    # For a given N, the number of representations is the number of powerful
    # numbers a in [1, N/2] such that N-a is also powerful.
    
    # Candidate N selection: try N values near the "center" of the powerful
    # number distribution, and also try N values that are sums of small powerful
    # numbers (which tend to be more frequent).
    
    # Generate candidate N values:
    # 1. Sums of the first K powerful numbers (small ones are more "compatible")
    # 2. Values near 2 * (median powerful number)
    
    # First, let's try a direct approach: sample many N values and count.
    # We'll focus on N in a range where both a and b can be powerful.
    
    # The maximum N we can represent is 2 * limit.
    # Let's try N values in a strategic range.
    
    # Approach: use the fact that for N, we need powerful numbers in [N-limit, limit]
    # (since both a and b must be <= limit). So N must be in [2, 2*limit].
    
    # For efficiency, we'll try candidate N values by:
    # 1. Taking pairs of small powerful numbers and computing their sums
    # 2. Counting frequencies of these sums
    # 3. Testing the most frequent sums
    
    # Sample sums from a subset of powerful numbers
    # Use the first 5000 powerful numbers (smallest ones) to generate candidate sums
    sample_size = min(5000, n_powerful)
    sample = powerful_list[:sample_size]
    
    # Generate candidate N values from sums of sample elements
    # Use a dictionary to count frequencies (but limit the number of unique sums)
    sum_counts = {}
    # Only generate sums from a smaller subset to keep it manageable
    sample2_size = min(2000, sample_size)
    sample2 = sample[:sample2_size]
    
    for i in range(sample2_size):
        a = sample2[i]
        for j in range(i, sample2_size):
            b = sample2[j]
            s = a + b
            if s <= 2 * limit:
                if s in sum_counts:
                    sum_counts[s] += 1
                else:
                    sum_counts[s] = 1
    
    # Get top candidate N values by frequency
    if sum_counts:
        top_candidates = sorted(sum_counts.items(), key=lambda x: -x[1])[:200]
        candidate_Ns = [s for s, c in top_candidates]
    else:
        candidate_Ns = []
    
    # Also add some strategic N values
    # N near the center of the range
    mid_N = limit
    for offset in range(0, 100000, 1000):
        candidate_Ns.append(mid_N + offset)
        candidate_Ns.append(mid_N - offset)
    
    # Also try N values that are multiples of small numbers (more likely to have representations)
    for mult in range(1, 100):
        candidate_Ns.append(mult * 10000)
    
    # Deduplicate and filter
    candidate_Ns = list(set(n for n in candidate_Ns if 2 <= n <= 2 * limit))
    
    # Sort candidates by a heuristic: prefer N values where the interval [N-limit, limit]
    # contains many powerful numbers
    # For now, just sort by value and test a subset
    
    # Test candidate N values, starting with those from sum_counts (most promising)
    tested = set()
    for N in candidate_Ns:
        if time.time() > deadline - 20:
            break
        if N in tested:
            continue
        tested.add(N)
        
        # Count representations for N
        # Only consider a in powerful_list where a <= N/2 and N-a <= limit
        a_min = max(1, N - limit)
        a_max = min(N // 2, limit)
        
        # Binary search to find the range in powerful_list
        left = bisect_right(powerful_list, a_min - 1)
        right = bisect_right(powerful_list, a_max)
        
        count = 0
        for idx in range(left, right):
            a = powerful_list[idx]
            b = N - a
            if b in powerful_set:
                count += 1
        
        if count > best_count:
            best_count = count
            # Generate the actual pairs
            pairs = []
            for idx in range(left, right):
                a = powerful_list[idx]
                b = N - a
                if b in powerful_set:
                    pairs.append([a, b])
            best_result = {'N': N, 'pairs': pairs}
    
    # If we haven't found anything, do a broader search
    if best_result is None or best_count < 5:
        # Try more N values in a sweep
        step = max(1, limit // 5000)
        for N in range(2 * step, 2 * limit, step):
            if time.time() > deadline - 15:
                break
            if N in tested:
                continue
            tested.add(N)
            
            a_min = max(1, N - limit)
            a_max = min(N // 2, limit)
            
            left = bisect_right(powerful_list, a_min - 1)
            right = bisect_right(powerful_list, a_max)
            
            if right - left > 100000:
                # Too many to check, skip or sample
                continue
            
            count = 0
            for idx in range(left, right):
                a = powerful_list[idx]
                b = N - a
                if b in powerful_set:
                    count += 1
            
            if count > best_count:
                best_count = count
                pairs = []
                for idx in range(left, right):
                    a = powerful_list[idx]
                    b = N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': N, 'pairs': pairs}
    
    if best_result is None:
        # Final fallback
        limit_small = 100_000
        powerful_small = _generate_powerful(limit_small)
        powerful_list_small = sorted(powerful_small)
        powerful_set_small = powerful_small
        arr = np.zeros(limit_small + 1, dtype=np.float64)
        for p in powerful_small:
            arr[p] = 1.0
        fft_size = 1
        while fft_size < 2 * (limit_small + 1):
            fft_size *= 2
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real
        max_idx = int(np.argmax(conv[2:])) + 2
        N_best = max_idx
        conv_val = int(round(conv[N_best]))
        if N_best % 2 == 0 and N_best // 2 <= limit_small and arr[N_best // 2] > 0:
            n_pairs = (conv_val + 1) // 2
        else:
            n_pairs = conv_val // 2
        pairs = []
        for a in powerful_list_small:
            if 2 * a > N_best:
                break
            b = N_best - a
            if b in powerful_set_small:
                pairs.append([a, b])
        best_result = {'N': N_best, 'pairs': pairs}
    
    return best_result
