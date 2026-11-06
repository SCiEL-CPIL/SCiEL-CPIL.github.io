import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int32)
    N = len(primes)

    if N == 0:
        return []

    INF = 10**9
    # Upper bound on sequence length: gaps are at least 1, so max length is bounded
    # by 1 + (10000-2)/1 but realistically much smaller. Use 100 as safe bound.
    max_possible_len = 100

    # dp[i][l] = min last gap for sequence of length l ending at prime i
    dp = np.full((N, max_possible_len + 1), INF, dtype=np.int32)
    parent = np.full((N, max_possible_len + 1), -1, dtype=np.int32)
    max_len = np.ones(N, dtype=np.int32)

    dp[:, 1] = 0

    best_len = 1
    best_index = 0

    for i in range(1, N):
        if time.time() > deadline:
            break

        pi = int(primes[i])
        # Vectorized: compute gaps from all previous primes to prime i
        gaps = pi - primes[:i]  # shape (i,)

        for l in range(1, max_possible_len):
            # For length l, check which j have dp[j][l] <= gap
            # dp[:, l] has shape (N,), we only care about j < i
            dp_prev = dp[:i, l]  # shape (i,)
            valid = dp_prev <= gaps  # boolean array
            if not np.any(valid):
                break  # No valid predecessor for this or larger lengths
            
            # Among valid j, find the one with minimum gap (max prime)
            valid_indices = np.where(valid)[0]
            min_gap_idx = valid_indices[np.argmin(gaps[valid_indices])]
            min_gap = gaps[min_gap_idx]
            
            if min_gap < dp[i, l + 1]:
                dp[i, l + 1] = min_gap
                parent[i, l + 1] = min_gap_idx
                if l + 1 > max_len[i]:
                    max_len[i] = l + 1

        if max_len[i] > best_len:
            best_len = max_len[i]
            best_index = i

    # Reconstruct
    if best_len == 1:
        return [int(primes[best_index])]

    path = []
    curr_idx = best_index
    curr_len = best_len

    while curr_len > 0:
        path.append(int(primes[curr_idx]))
        if curr_len == 1:
            break
        prev_idx = int(parent[curr_idx, curr_len])
        curr_idx = prev_idx
        curr_len -= 1

    path.reverse()
    return path
