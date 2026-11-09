"""DP over prime subsequences: maximize length with nondecreasing successive gaps.

Uses a prefix-maximum array so each transition is O(1):
  dp[i][g]  = max length of valid sequence ending at prime i with last gap exactly g
  pref[i][g] = max(dp[i][0], ..., dp[i][g])
Transition for prime i, previous prime j, gap g = primes[i]-primes[j]:
  dp[i][g] = max(dp[i][g], pref[j][g] + 1)
"""
import time
import numpy as np

_DEADLINE = time.time() + 85.0


def construct():
    global _DEADLINE
    deadline = _DEADLINE
    LIMIT = 10000

    # Sieve primes up to LIMIT using NumPy.
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for x in range(2, int(LIMIT ** 0.5) + 1):
        if sieve[x]:
            sieve[x * x::x] = False
    primes = np.array([i for i in range(2, LIMIT + 1) if sieve[i]], dtype=np.int32)
    N = len(primes)

    # Maximum possible gap between two primes <= LIMIT is LIMIT - 2.
    G = LIMIT - 2

    # dp[i][g]: max length of valid sequence ending at prime i with last gap exactly g.
    # Use -1 to indicate "unreachable" so that pref[j][g] correctly reflects
    # the best reachable predecessor state.
    dp = np.full((N, G + 1), -1, dtype=np.int32)
    pref = np.full((N, G + 1), -1, dtype=np.int32)
    pred = np.full((N, G + 1), -1, dtype=np.int32)

    # Base case: a single prime is a valid sequence of length 1 with no gap.
    # We model "no gap yet" as gap 0 with length 1.
    dp[:, 0] = 1
    pref[:, 0] = 1

    best_len = 1
    best_i = 0
    best_g = 0

    for i in range(1, N):
        if time.time() > deadline:
            break
        pi = primes[i]

        # For each previous prime j, compute gap and update dp[i][gap].
        for j in range(i):
            gap = int(pi - primes[j])
            # Best sequence ending at j with last gap <= gap:
            #   - if gap == 0, only the base state (length 1) applies
            #   - otherwise, pref[j][gap] gives the max length over gaps 0..gap
            if gap == 0:
                val = 1 + 1  # extend base state (length 1) -> length 2
            else:
                best_prev = pref[j][gap]
                if best_prev < 0:
                    continue
                val = best_prev + 1
            if val > dp[i][gap]:
                dp[i][gap] = val
                pred[i][gap] = j

        # Compute prefix maximums for row i (over gaps 0..G).
        # np.maximum.accumulate gives pref[i][g] = max(dp[i][0..g]).
        pref[i] = np.maximum.accumulate(dp[i])

        # Update global best.
        row_max = int(np.max(dp[i]))
        if row_max > best_len:
            best_len = row_max
            best_g = int(np.argmax(dp[i]))
            best_i = i

    # Reconstruct the sequence by walking predecessors.
    seq = []
    if best_len > 0:
        i = best_i
        g = best_g
        while i >= 0:
            seq.append(int(primes[i]))
            if i == 0:
                break
            j = int(pred[i][g])
            if j == -1:
                break
            # Find the specific gap h at predecessor j such that dp[j][h] == pref[j][g]
            # and h <= g. We walk down from g to 0.
            target_val = int(pref[j][g])
            h = g
            while h >= 0:
                if int(dp[j][h]) == target_val:
                    break
                h -= 1
            if h < 0:
                # Should not happen; fall back to g.
                h = g
            i = j
            g = h
        seq.reverse()

    # Safety fallback: if reconstruction produced nothing, use a simple greedy.
    if not seq:
        from math import isqrt
        vals = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(vals) < 2 or p - vals[-1] >= vals[-1] - vals[-2]:
                    vals.append(p)
        seq = vals

    return seq
