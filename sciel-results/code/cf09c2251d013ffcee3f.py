"""Longest prime subsequence with nondecreasing gaps, via DP + Fenwick tree."""
import time

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s wall-clock

    # 1) Sieve primes up to 10000
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    n = len(primes)

    # 2) Coordinate-compress gaps. Gap between primes[i-1] and primes[i].
    #    For a sequence ending at index i, the "last gap" is primes[i] - primes[prev].
    #    We need, for each i, the max length among predecessors j < i with gap_j <= (primes[i]-primes[j]).
    #    But gap depends on j, so we can't precompute a single gap per i.
    #
    #    Alternative formulation: state = (last_prime_index, last_gap).
    #    dp[i][g] too large. Instead: for each i, consider all j<i, gap = primes[i]-primes[j].
    #    We want max over j<i of (dp ending at j with last_gap <= gap) + 1.
    #    This is O(n^2) if we do it naively. n ~ 1229, so n^2 ~ 1.5M, fine.
    #
    #    But to use Fenwick: we can process i from 0..n-1. For each i, we need to query
    #    "among all j < i, what is the max sequence length ending at j whose last_gap <= gap_ij?"
    #    The last_gap for a sequence ending at j is primes[j] - primes[prev_j], which varies.
    #
    #    Simpler: let dp[i] = best length of a valid subsequence ending at primes[i],
    #    and let lastgap[i] = the gap used to reach i (primes[i] - primes[prev]).
    #    For a new i, we need max over j < i with lastgap[j] <= (primes[i]-primes[j]) of dp[j].
    #    This still depends on j in the gap. O(n^2) is acceptable for n~1229.

    # O(n^2) DP with backtracking
    # dp[i] = length of longest valid subsequence ending at index i
    # prev[i] = predecessor index
    # gap[i] = primes[i] - primes[prev[i]] (the gap into i)
    INF = 10**9
    dp = [1] * n
    prev = [-1] * n
    gap = [0] * n  # gap into i

    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() > deadline:
            break
        pi = primes[i]
        # Try all j < i as predecessor
        best_j = -1
        best_dp = 0
        for j in range(i):
            gj = gap[j]  # last gap of sequence ending at j
            gij = pi - primes[j]
            if gj <= gij:
                if dp[j] > best_dp:
                    best_dp = dp[j]
                    best_j = j
        if best_j >= 0:
            dp[i] = best_dp + 1
            prev[i] = best_j
            gap[i] = pi - primes[best_j]
        if dp[i] > best_len:
            best_len = dp[i]
            best_end = i

    # Reconstruct
    seq = []
    cur = best_end
    while cur >= 0:
        seq.append(primes[cur])
        cur = prev[cur]
    seq.reverse()

    return seq
