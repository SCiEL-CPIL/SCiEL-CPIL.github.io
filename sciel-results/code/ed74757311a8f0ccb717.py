import time
from math import isqrt
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def time_left():
        return deadline - time.time()

    # Generate powerful numbers up to M
    # n = a^2 * b^3, b square-free
    # We want M large enough to get many pairs, but K (count of powerfuls)
    # manageable for the counting algorithm.
    
    # Let's target K ~ 20,000.
    # K ~ C * M^(2/5) => M ~ (K/C)^(5/2)
    # If K=20000, M ~ 10^11
    
    M = 50_000_000_000  # 5 * 10^10
    
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    def is_square_free(n):
        if n < 2:
            return n == 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % (p * p) == 0:
                return False
            while temp % p == 0:
                temp //= p
            p += 1 if p == 2 else 2
        return True

    powerful = set()
    
    # Optimization: Precompute square-free b's
    # Or just loop and check. max_b is ~ 3700.
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
            
    if time.time() > deadline - 30:
        # Fallback to smaller M if generation is slow (unlikely)
        M = 5_000_000
        powerful = set()
        max_b = int(M ** (1.0 / 3.0)) + 2
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b_cubed = b ** 3
            if b_cubed > M:
                break
            max_a = isqrt(M // b_cubed)
            for a in range(1, max_a + 1):
                powerful.add(a * a * b_cubed)

    powerful_list = sorted(powerful)
    K = len(powerful_list)
    P = np.array(powerful_list, dtype=np.int64)
    powerful_set = powerful

    # Count pairs
    # If K is small, use full outer product
    # If K is large, use chunked unique
    
    if K <= 5000:
        # Full outer product
        # sums = np.add.outer(P, P)
        # This creates a K x K array.
        # For K=5000, 25M elements. 200MB. OK.
        sums = np.add.outer(P, P)
        # We need counts of each sum.
        # np.bincount requires array size up to max sum.
        # Max sum is 2*M.
        # If M is large, bincount array is huge.
        # Use np.unique instead.
        vals, counts = np.unique(sums.ravel(), return_counts=True)
        max_count = np.max(counts)
        best_N = vals[np.argmax(counts)]
    else:
        # Chunked approach
        global_counts = defaultdict(int)
        chunk_size = 100
        for i in range(0, K, chunk_size):
            end_i = min(i + chunk_size, K)
            chunk_P = P[i:end_i]
            # We only need j >= i to avoid double counting?
            # No, np.add.outer(P[i:end_i], P[i:]) gives pairs (a,b) with a from chunk, b from P[i:].
            # This covers all pairs (a,b) with index(a) >= i and index(b) >= i.
            # Wait, if we iterate i, we miss pairs where index(a) < i and index(b) >= i?
            # Yes.
            # We need to count all pairs (a,b) with a<=b.
            # Standard trick:
            # Iterate i from 0 to K-1.
            # For each i, consider pairs (P[i], P[j]) for j >= i.
            # This is O(K^2/2).
            # Chunking:
            # For a chunk of 'a's (indices i to i+C-1), we need to sum with 'b's (indices i to K-1, i+1 to K-1, etc).
            # This is complex to chunk correctly for unique pairs.
            
            # Simpler: Just do all pairs (a,b) where a,b in P.
            # Since a+b = b+a, we can just count all ordered pairs and divide by 2?
            # No, a=a is counted once.
            # If we count all ordered pairs (a,b) from P x P, then:
            # Count(N) = sum_{a+b=N} 1.
            # This includes (a,b) and (b,a).
            # If a != b, they are distinct.
            # If a == b, it's one pair.
            # The problem asks for distinct pairs a <= b.
            # So if we count ordered pairs, let C_ord(N) be the count.
            # Then C_unord(N) = (C_ord(N) + I(N even and N/2 in P)) / 2.
            # This is easier to chunk!
            
            # Let's restart the counting logic for K > 5000.
            # We will count ordered pairs.
            
            global_counts = defaultdict(int)
            chunk_size = 200
            for i in range(0, K, chunk_size):
                end_i = min(i + chunk_size, K)
                chunk_P = P[i:end_i]
                # Sum with ALL P
                sums = np.add.outer(chunk_P, P)
                vals, counts = np.unique(sums.ravel(), return_counts=True)
                for v, c in zip(vals, counts):
                    global_counts[v] += c
                if time.time() > deadline - 10:
                    break
                    
            if time.time() > deadline - 5:
                # Timeout, return best so far?
                # We need to finish the loop to get the best.
                # If we break, we have partial counts.
                # This is risky.
                pass

            # Convert ordered counts to unordered
            best_score = 0
            best_N = -1
            for N, c_ord in global_counts.items():
                n_eq = 0
                if N % 2 == 0:
                    half = N // 2
                    if half in powerful_set:
                        n_eq = 1
                score = (c_ord + n_eq) // 2
                if score > best_score:
                    best_score = score
                    best_N = N

    if best_N == -1:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Reconstruct pairs
    pairs = []
    limit = best_N // 2
    end_idx = np.searchsorted(P, limit)
    for i in range(end_idx):
        a = int(P[i])
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}
