"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via DP on pairs."""
import time
import numpy as np
from math import isqrt

def construct():
    deadline = time.time() + 78  # safety margin before 90s kill
    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, isqrt(N) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = 0
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    P = len(primes)
    if P < 2:
        return primes

    # dp[i][j] for i<j: max length ending ...p_i,p_j. Use list of lists.
    # dp[i][j] stored in a 2D list; dp[i][j] = 2 baseline.
    dp = [[2] * P for _ in range(P)]  # dp[i][j] only valid for i<j
    # Two-pointer optimization: for fixed i, as j increases, T=2*pi-primes[j] decreases,
    # so k_min is non-increasing. We track k_min and only decrement it.
    best_len = 2
    best_pair = (0, 1)  # indices
    primes_arr = np.array(primes)
    for i in range(1, P):
        # build suffix max over k in [0, i-1] of dp[k][i]
        # suffix[i-1] = dp[i-1][i]; suffix[m] = max(dp[m][i], suffix[m+1])
        col = np.array([dp[m][i] for m in range(i)])
        # suffix max from right to left
        suff = np.maximum.accumulate(col[::-1])[::-1]
        # suff[m] = max(dp[m][i], dp[m+1][i], ..., dp[i-1][i])
        pi = primes[i]
        k_min = 0  # will be adjusted
        for j in range(i + 1, P):
            T = 2 * pi - primes[j]
            # find smallest k in [0, i-1] with primes[k] >= T
            # Since T is non-increasing as j increases, k_min is non-increasing
            # Adjust k_min downward while primes[k_min-1] >= T
            while k_min > 0 and primes[k_min - 1] >= T:
                k_min -= 1
            # Now k_min is the smallest index with primes[k_min] >= T
            # But we need to verify: if primes[k_min] < T, then no valid k
            if k_min < i and primes[k_min] >= T:
                cand = 1 + int(suff[k_min])
            else:
                # Check if any k in [0,i-1] satisfies primes[k] >= T
                # Since primes is sorted, if primes[i-1] < T then no valid k
                if i > 0 and primes[i - 1] >= T:
                    # k_min should have been adjusted; this shouldn't happen
                    cand = 1 + int(suff[0])
                else:
                    cand = 2
            dp[i][j] = cand
            if cand > best_len:
                best_len = cand
                best_pair = (i, j)
        if time.time() > deadline - 10:
            break
    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes
    # backtrack
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        # find k < i with dp[k][i] + 1 == dp[i][j] and primes[i]-primes[k] <= primes[j]-primes[i]
        target = dp[i][j] - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if dp[k][i] == target and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            break
    seq.reverse()
    # Verify the sequence is valid: increasing primes with nondecreasing gaps
    if len(seq) >= 2:
        gaps = [seq[m+1] - seq[m] for m in range(len(seq)-1)]
        for m in range(len(gaps)-1):
            if gaps[m+1] < gaps[m]:
                # Invalid; fall back to first two
                return seq[:2]
    return seq
