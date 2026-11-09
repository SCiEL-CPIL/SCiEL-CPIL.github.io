import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit.
        A powerful number has the form u^2 * v^3 where v is square-free.
        """
        powerful = set()
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(round(limit ** (1.0 / 3.0))) + 1
        
        # Generate square-free numbers up to max_v
        sq_free = [False] * (max_v + 1)
        sq_free[0] = True  # 0 is not square-free (or we don't care)
        sq_free[1] = True  # 1 is square-free
        for i in range(2, max_v + 1):
            sq_free[i] = True
        
        for p in range(2, int(math.isqrt(max_v)) + 1):
            p2 = p * p
            for j in range(p2, max_v + 1, p2):
                sq_free[j] = False
        
        sq_free_list = [v for v in range(1, max_v + 1) if sq_free[v]]
        
        for v in sq_free_list:
            v3 = v ** 3
            if v3 > limit:
                continue
            # u^2 * v^3 <= limit => u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        
        return sorted(powerful)

    # Generate powerful numbers up to a large limit
    # The number of powerful numbers up to M is roughly M^(1/3) + M^(1/4) + ...
    # For M = 10^7, this is manageable (a few thousand to tens of thousands)
    # For M = 10^8, it's larger but still feasible
    # We want to maximize the number of pairs, so we want a large set of powerful numbers
    # but also N should be such that many pairs sum to it.
    
    # Let's try M = 2 * 10^7 to get a good density
    M = 20_000_000
    P = generate_powerful(M)
    P_set = set(P)
    P_arr = np.array(P, dtype=np.int64)
    L = len(P)
    
    if L == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_count = 0
    best_N = 2
    best_pairs = [[1, 1]]
    
    # Strategy: Use the fact that for a fixed N, the number of representations
    # is the number of a in P such that N - a is also in P and a <= N - a.
    # 
    # To find N with many representations, we can look at the "middle" of the range
    # where both a and b = N - a are likely to be powerful.
    # 
    # A good heuristic: N values that are sums of many small powerful numbers
    # will tend to have more representations.
    #
    # Better approach: Use convolution-like counting.
    # For each pair (a, b) with a <= b, a + b = N. We want to count how many
    # such pairs exist for each N.
    #
    # Since P is sorted, for a given N, we can use two pointers:
    # a starts from smallest, b starts from largest, move them toward each other.
    # This is O(L) per N.
    #
    # But we don't want to check all N up to 2*M. Instead, we generate candidate
    # N values from sums of pairs of powerful numbers, then verify.
    #
    # Even better: use numpy to compute all pair sums and count frequencies.
    # But L can be ~10^4-10^5, so L^2 is too large.
    #
    # Compromise: take a subset of powerful numbers (the smaller ones, which are
    # more densely packed) and compute their pair sums.
    
    # Take the first K powerful numbers (smallest ones)
    K = min(5000, L)
    small_P = P[:K]
    small_P_arr = np.array(small_P, dtype=np.int64)
    
    # Compute all pair sums a + b where a <= b from small_P
    # This gives us candidate N values with their "small pair" counts
    # Use broadcasting for efficiency
    # small_P_arr[:, None] + small_P_arr[None, :] gives all sums
    # But we only want a <= b, so upper triangle
    
    # For memory efficiency, do it in chunks
    chunk_size = 1000
    sum_counts = np.zeros(2 * M + 1, dtype=np.int32)
    
    for i_start in range(0, K, chunk_size):
        i_end = min(i_start + chunk_size, K)
        chunk = small_P_arr[i_start:i_end]
        # For each a in chunk, b ranges from a to small_P_arr[-1]
        # Actually, for a <= b, b >= a
        # sums = chunk[:, None] + small_P_arr[None, :]
        # But we only want b >= a, i.e., small_P_arr >= chunk
        # This is complex. Let's just do all pairs and divide by 2 for a < b,
        # and handle a == b separately.
        
        # Simpler: just compute all sums for this chunk
        sums = chunk[:, None] + small_P_arr[None, :]
        # Flatten and bincount
        flat_sums = sums.flatten()
        # Only count sums <= 2*M
        valid = flat_sums <= 2 * M
        if np.any(valid):
            flat_sums = flat_sums[valid]
            # bincount
            counts = np.bincount(flat_sums, minlength=2 * M + 1)
            sum_counts += counts.astype(np.int32)
    
    # Find top candidate N values
    # We want N in [2, 2*M]
    top_candidates = np.argsort(sum_counts)[::-1]
    
    # For each candidate N, verify by counting actual pairs in the full P set
    max_candidates_to_try = 200
    
    for idx in range(min(max_candidates_to_try, len(top_candidates))):
        if time.time() > deadline - 20:
            break
        
        N = int(top_candidates[idx])
        if N < 2 or N > 2 * M:
            continue
        
        # Count pairs (a, b) with a <= b, a + b = N, both in P
        # Use two pointers on P (sorted)
        pairs = []
        i = 0
        j = L - 1
        while i <= j:
            s = P[i] + P[j]
            if s == N:
                pairs.append([P[i], P[j]])
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        
        c = len(pairs)
        if c > best_count:
            best_count = c
            best_N = N
            best_pairs = pairs
    
    # If the top candidates didn't yield good results, try a broader search
    # on a grid of N values
    if best_count < 10:
        # Try N values in the range [M//2, 3*M//2] with some step
        step = max(1, M // 500)
        for N in range(M // 2, min(3 * M // 2, 2 * M) + 1, step):
            if time.time() > deadline - 10:
                break
            
            pairs = []
            i = 0
            j = L - 1
            while i <= j:
                s = P[i] + P[j]
                if s == N:
                    pairs.append([P[i], P[j]])
                    i += 1
                    j -= 1
                elif s < N:
                    i += 1
                else:
                    j -= 1
            
            c = len(pairs)
            if c > best_count:
                best_count = c
                best_N = N
                best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
