import numpy as np
import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def sieve(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        if limit >= 0:
            is_prime[0] = 0
        if limit >= 1:
            is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit+1) if is_prime[i]]

    primes = sieve(1_000_000)

    def check_powerful(n):
        if n <= 1:
            return True
        rem = n
        for p in primes:
            if p * p > rem:
                break
            if rem % p == 0:
                cnt = 0
                while rem % p == 0:
                    rem //= p
                    cnt += 1
                if cnt == 1:
                    return False
        if rem > 1:
            r = math.isqrt(rem)
            if r * r == rem:
                return True
            # Check perfect cube
            c = int(round(rem ** (1.0/3.0)))
            for cc in (c-1, c, c+1, c+2):
                if cc > 0 and cc * cc * cc == rem:
                    return True
            return False
        return True

    def generate_powerful(M):
        """Generate all powerful numbers <= M."""
        pw = set()
        # Powerful numbers: u^2 * v^3 where v^3 <= M
        max_v = int(M ** (1.0/3.0)) + 2
        for v in range(1, max_v + 1):
            v3 = v * v * v
            if v3 > M:
                break
            max_u = math.isqrt(M // v3)
            for u in range(1, max_u + 1):
                val = u * u * v3
                if val <= M:
                    pw.add(val)
        return sorted(pw)

    def count_pairs_chunked(P, N_cand, P_set):
        """Count pairs (a,b) in P with a<=b, a+b=N_cand. Returns list of pairs."""
        pairs = []
        n = len(P)
        # Binary search for upper bound: a <= N_cand//2
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if P[mid] <= N_cand // 2:
                lo = mid + 1
            else:
                hi = mid
        count_a = lo  # number of valid a values
        # Vectorized: for all a in P[:count_a], check if N_cand - a is in P_set
        a_arr = P[:count_a]
        b_arr = N_cand - a_arr
        # Check membership
        mask = np.array([b in P_set for b in b_arr], dtype=bool)
        valid_a = a_arr[mask]
        valid_b = b_arr[mask]
        # Ensure a <= b
        final_mask = valid_a <= valid_b
        pairs = [[int(a), int(b)] for a, b in zip(valid_a[final_mask], valid_b[final_mask])]
        return pairs

    def find_best_in_range(P, P_set, time_budget):
        """
        Use chunked pairwise sum counting to find N with maximum pairs.
        Strategy: sample sum values densely, count occurrences.
        """
        K = len(P)
        if K < 2:
            return 0, 0, []

        # Use dictionary-based counting with chunked vectorization
        # For each chunk of a-values, compute all sums with b >= a
        # This is O(K^2) but vectorized per chunk

        # Better approach: use the fact that we want the MAX count.
        # Sample N values and count.
        # The densest region is around the median of pair sums.
        # Pair sums range from P[0]+P[0] to P[-1]+P[-1].
        # For a set of size K, the number of distinct sums is at most K^2/2.
        # The maximum multiplicity is at most K/2 (when a+b is constant for many a).

        # Strategy: iterate over a in P, for each a compute b = N - a for candidate N.
        # But we don't know N. Instead, use a two-pointer / meet-in-middle approach.

        # Most effective: build a frequency array via convolution-like counting.
        # Since K can be up to ~50k, K^2 = 2.5 billion - too slow.
        # Use FFT for the counting, then verify top candidates.

        # Actually, let's use a smarter sampling approach:
        # The maximum number of pairs for a given N is at most K/2.
        # For random-like sets, the expected max is much smaller.
        # We'll use FFT to find top candidates, then verify.

        # FFT approach for counting
        max_sum = P[-1] + P[-1]
        # Use a smaller FFT size if possible
        fft_size = 1
        while fft_size < max_sum + 1:
            fft_size *= 2

        # For very large fft_size, use chunked approach
        if fft_size > 2**28:  # ~268M, too large for float32 array
            # Use chunked counting
            return _chunked_search(P, P_set, time_budget)
        else:
            arr = np.zeros(fft_size, dtype=np.float32)
            arr[P] = 1.0
            fft_arr = np.fft.rfft(arr)
            conv = np.fft.irfft(fft_arr * fft_arr, n=fft_size)

            # Find top candidates
            top_k = min(200, len(conv))
            top_indices = np.argpartition(conv, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]

            best_N = 0
            best_count = 0
            best_pairs = []

            for idx in top_indices:
                N_cand = int(idx)
                if N_cand < 2:
                    continue
                if time.time() > time_budget:
                    break

                pairs = count_pairs_chunked(P, N_cand, P_set)
                if len(pairs) > best_count:
                    best_count = len(pairs)
                    best_N = N_cand
                    best_pairs = pairs

                # Early termination: if conv value can't beat current best
                if conv[idx] < best_count:
                    break

            return best_N, best_count, best_pairs

    def _chunked_search(P, P_set, time_budget):
        """
        Chunked pairwise sum counting for very large P.
        Split P into chunks, for each pair of chunks compute sums.
        """
        K = len(P)
        chunk_size = 2000  # 2000x2000 = 4M operations per chunk pair

        # We'll count sums using a dictionary
        # But K^2/2 is too large. Instead, use the observation that
        # for the maximum, we need many a's to map to the same N.
        # Use random sampling of N values near the center of the sum distribution.

        # The sum distribution is roughly triangular, centered at (P[0]+P[-1])
        center = (P[0] + P[-1]) // 2

        # Sample N values around the center
        # The width of the dense region is roughly K * (P[-1]-P[0])/K = P[-1]-P[0]
        # But the peak density is at the center.
        # Sample a range of N values.

        # Actually, for a set of K powerful numbers, the number of pairs summing to N
        # is maximized when N is such that N-P contains many elements of P.
        # This is a correlation problem.

        # Use a different approach: for each a in P, the set of valid b's is P ∩ [a, N-a].
        # For a fixed N, count = |P ∩ [a, N-a]| summed over... no, it's just |{a in P : N-a in P, a<=N-a}|.

        # Let's just do a direct but optimized search:
        # For the top candidates, we know from theory that N should be even
        # (since 1 is powerful, and many powerful numbers are even).

        # Use FFT with a smaller effective range by shifting:
        # Only consider P values in [P_min, P_max] where the sums fall in a reasonable range.
        # Actually, just use the full FFT but with float64 for accuracy.

        # For K up to 50k, the FFT size is ~10^9 which is too large.
        # Use the "sparse" approach: iterate over a, compute all b = N - a for a range of N.

        # Best approach for large K: use the fact that powerful numbers have special structure.
        # a = u^2 * v^3. For fixed v, a ranges over u^2 * v^3 for u = 1, 2, ..., max_u.
        # So for fixed v, the set of a's is {v^3, 4v^3, 9v^3, ...}.
        # Similarly for b = u'^2 * v'^3.
        # N = v^3 * u^2 + v'^3 * u'^2.
        # For fixed v, v', this is a quadratic Diophantine equation.

        # Let's use a hybrid: FFT for the counting (with appropriate size), verify top.
        # For K=50k, max_sum ~ 10^9, FFT size ~ 2^30 = 1B. Too large.
        # Use rfft with n = next_power_of_2(max_sum+1). For max_sum=10^9, that's 2^30.
        # 2^30 float32 = 4GB. Too much.

        # Alternative: use multiple smaller FFTs (chunked convolution).
        # Or: use the O(K^2) approach but only for a subset.

        # Practical approach: use a dictionary and iterate, but with early termination.
        # For K=50k, K^2/2 = 1.25B. Too slow in Python.

        # Use NumPy vectorization in chunks:
        # For a chunk of a-values (size C), compute all sums with all b-values.
        # C * K operations per chunk. With C=1000, K=50000: 50M per chunk.
        # Number of chunks: K/C = 50. Total: 2.5B. Still too slow.

        # Better: for each a, the number of b's that give a "new" N is at most K.
        # But we only care about the MAX count.
        # Use the following: the maximum possible count is K/2.
        # In practice, for random-like sets, the max is O(log K) or O(K^0.5).
        # For powerful numbers, the structure might allow higher counts.

        # Let's use a targeted search:
        # 1. Generate all sums for a subset of P (e.g., first 10k elements).
        # 2. Find the most frequent sum.
        # 3. Verify with the full P.
        # 4. Also try other subsets.

        subset_size = min(10000, K)
        P_sub = P[:subset_size]
        P_sub_set = set(P_sub)

        # Count sums within the subset
        counts = defaultdict(int)
        for i in range(len(P_sub)):
            a = P_sub[i]
            if time.time() > time_budget:
                break
            # For each a, count how many b in P_sub give the same sum
            # Vectorized: b_arr = N - a for candidate N... but we don't know N.
            # Just do pairwise:
            b_arr = P_sub[i:]  # b >= a
            sums = a + b_arr
            # Count frequencies
            # This is O(K^2) for the subset = 10k^2/2 = 50M. Doable in ~5s with numpy.
            # But we need to find the max, not all counts.
            # Use np.unique with return_counts:
            unique_sums, counts_arr = np.unique(sums, return_counts=True)
            max_idx = np.argmax(counts_arr)
            if counts_arr[max_idx] > 0:
                N_cand = int(unique_sums[max_idx])
                # Verify with full P
                pairs = count_pairs_chunked(P, N_cand, P_set)
                if len(pairs) > 0:
                    return N_cand, len(pairs), pairs

        return 0, 0, []

    # Main search strategy:
    # 1. Generate powerful numbers up to M.
    # 2. Use FFT (if feasible) or chunked search to find best N.
    # 3. Verify with exact factorization.

    global_best_N = 2
    global_best_pairs = [[1, 1]]
    global_best_count = 1

    # Try multiple M values, increasing until time runs out
    M_values = [10_000_000, 50_000_000, 100_000_000, 200_000_000, 400_000_000, 500_000_000]

    for M in M_values:
        if time.time() > deadline - 20:
            break

        P = generate_powerful(M)
        K = len(P)
        P_set = set(P)

        if K < 2:
            continue

        # Determine if FFT is feasible
        max_sum = P[-1] + P[-1]
        fft_size = 1
        while fft_size < max_sum + 1:
            fft_size *= 2

        time_budget = deadline - 15

        if fft_size <= 2**27:  # ~134M, float32 = 512MB
            N_cand, count, pairs = find_best_in_range(P, P_set, time_budget)
        else:
            N_cand, count, pairs = _chunked_search(P, P_set, time_budget)

        if count > global_best_count:
            global_best_count = count
            global_best_N = N_cand
            global_best_pairs = pairs

        # Verify all pairs with exact factorization
        if global_best_pairs:
            verified_pairs = []
            for a, b in global_best_pairs:
                if a + b == global_best_N and a <= b and check_powerful(a) and check_powerful(b):
                    verified_pairs.append([a, b])
            global_best_pairs = verified_pairs
            global_best_count = len(verified_pairs)

        if time.time() > deadline - 10:
            break

    # Final safety: ensure we have a valid result
    if not global_best_pairs:
        global_best_N = 2
        global_best_pairs = [[1, 1]]
        global_best_count = 1

    return {'N': global_best_N, 'pairs': global_best_pairs}
