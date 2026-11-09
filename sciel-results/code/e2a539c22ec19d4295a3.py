"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import bisect
from collections import defaultdict

def _sieve(limit):
    """Return numpy array of all primes <= limit using a standard sieve."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

def construct():
    deadline = time.time() + 80.0
    t_start = time.time()

    # All primes <= 10000
    primes = _sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    p_list = primes.tolist()
    p_arr = primes

    # Group primes by their gap-to-previous-prime
    # gap_i = p[i] - p[i-1] for i >= 1
    gap_to_prev = np.diff(p_arr).astype(np.int64)  # length n-1, gap_to_prev[i] = p[i+1]-p[i]

    # For each prime index i (1..n-1), its "gap" is p[i] - p[i-1]
    # We'll use this as a key for grouping

    # DP formulation:
    # f[i] = length of longest valid subsequence ending at prime i
    # A valid subsequence has nondecreasing gaps.
    # If the subsequence ends at ...j, i, then gap(j,i) >= gap(prev,j)
    # where gap(j,i) = p[i] - p[j]
    #
    # Key insight: if we fix the last two elements as (j, i), then the gap is g = p[i]-p[j].
    # For the previous element k, we need p[j]-p[k] <= g.
    #
    # Alternative DP: Let dp[i] = longest valid subsequence ending at index i.
    # dp[i] = 1 + max{dp[j] : j < i and (if j is preceded by k in the subsequence, p[j]-p[k] <= p[i]-p[j])}
    # This is tricky because dp[j] alone doesn't tell us the gap into j.
    #
    # Better: dp[i][g] = longest valid subsequence ending at i with last gap exactly g.
    # But gaps can be large.
    #
    # Let's use: dp[i] = best length ending at i, and track the minimum last-gap that achieves it.
    # Actually, for a fixed endpoint i, among all valid subsequences ending at i, we want the one
    # with the smallest last gap (to allow more extensions). But we also want maximum length.
    #
    # Revised: For each i, we want to find the longest valid subsequence ending at i.
    # A valid subsequence ending at i has the form [..., j, i] where the gap p[i]-p[j] >= the gap
    # before j in the subsequence.
    #
    # Let's define: best[i] = (length, min_last_gap) where among all valid subsequences ending at i,
    # we track the maximum length, and for that length, the minimum possible last gap.
    #
    # Transition: best[i].length = 1 + max{best[j].length : j < i, p[i]-p[j] >= best[j].min_last_gap}
    # best[i].min_last_gap = p[i] - argmax_j (the j that achieved the max)
    # But if multiple j give the same length, pick the one with smallest p[i]-p[j] (i.e., largest j).
    #
    # This is O(n^2) in the worst case but with n=1229, that's ~1.5M operations, very fast.

    # Actually, let me reconsider. The constraint is that gaps are NONDECREASING.
    # So if the subsequence is p[a1], p[a2], ..., p[ak] with a1 < a2 < ... < ak,
    # then p[a2]-p[a1] <= p[a3]-p[a2] <= ... <= p[ak]-p[ak-1].
    #
    # DP: Let L[i] = length of longest valid subsequence ending at index i.
    # Let G[i] = the last gap in that subsequence (p[i] - p[j] where j is the second-to-last).
    #
    # For i, L[i] = 1 + max{L[j] : j < i and p[i]-p[j] >= G[j]}
    # G[i] = p[i] - j* where j* is the j that achieves the max (and among ties, the one with smallest p[i]-p[j])
    #
    # But this doesn't work directly because L[j] and G[j] are coupled.
    # A subsequence ending at j with length L[j] has last gap G[j]. For extension to i, we need p[i]-p[j] >= G[j].
    #
    # However, there might be a shorter subsequence ending at j with a smaller last gap that could be extended.
    # So we need to track, for each j, the Pareto-optimal (length, last_gap) pairs.
    #
    # Given the small n (1229), let's use a simpler O(n^2) approach:
    # For each i, for each j < i, check if we can extend a subsequence ending at j to i.
    # But we need to know the last gap of the subsequence ending at j.
    #
    # Let's use: dp[i] = list of (length, last_gap) Pareto-optimal pairs for subsequences ending at i.
    #
    # For i=0: dp[0] = [(1, 0)]
    # For i>0:
    #   Start with (1, 0)
    #   For each j < i:
    #     For each (l, g) in dp[j]:
    #       new_gap = p[i] - p[j]
    #       if new_gap >= g:
    #         new_pair = (l+1, new_gap)
    #         Add to candidates
    #   Pareto-optimize: remove dominated pairs (a pair (l1,g1) dominates (l2,g2) if l1>=l2 and g1<=g2)
    #
    # This could be expensive if dp[j] is large, but in practice it should be small.

    dp = [None] * n
    dp[0] = [(1, 0)]

    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() > deadline:
            break
        candidates = [(1, 0)]
        for j in range(i):
            gap_ij = p_list[i] - p_list[j]
            for (l, g) in dp[j]:
                if gap_ij >= g:
                    candidates.append((l + 1, gap_ij))
        # Pareto-optimize
        # Sort by length descending, then by gap ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        pareto = []
        max_len = 0
        min_gap_for_len = {}
        for (l, g) in candidates:
            if l > max_len:
                max_len = l
                min_gap_for_len = {l: g}
                pareto = [(l, g)]
            elif l == max_len:
                if g < min_gap_for_len[l]:
                    min_gap_for_len[l] = g
                    pareto = [(l, g)]
        # Actually, proper Pareto: keep (l,g) if no other (l',g') has l'>=l and g'<=g with at least one strict
        # Simpler: group by length, for each length keep min gap, then remove lengths that are dominated
        from collections import defaultdict
        len_to_mingap = defaultdict(lambda: float('inf'))
        for (l, g) in candidates:
            if g < len_to_mingap[l]:
                len_to_mingap[l] = g
        # Now for each length, we have the min gap. Remove dominated lengths.
        # A length l is dominated if there exists l' > l with mingap[l'] <= mingap[l]
        lengths_sorted = sorted(len_to_mingap.keys(), reverse=True)
        pareto = []
        best_gap = float('inf')
        for l in lengths_sorted:
            if len_to_mingap[l] < best_gap:
                pareto.append((l, len_to_mingap[l]))
                best_gap = len_to_mingap[l]
        # pareto is now sorted by length descending
        dp[i] = pareto
        if pareto and pareto[0][0] > best_len:
            best_len = pareto[0][0]
            best_end = i

    # Now we need to reconstruct the actual subsequence.
    # For this, we need to track the parent.
    # Let's redo with parent tracking.

    # dp2[i] = list of (length, last_gap, parent_j) Pareto-optimal
    dp2 = [None] * n
    dp2[0] = [(1, 0, -1)]

    best_len2 = 1
    best_end2 = 0

    for i in range(1, n):
        if time.time() > deadline:
            break
        candidates = [(1, 0, -1)]
        for j in range(i):
            gap_ij = p_list[i] - p_list[j]
            for (l, g, parent) in dp2[j]:
                if gap_ij >= g:
                    candidates.append((l + 1, gap_ij, j))
        # Pareto-optimize with parent tracking
        len_to_best = {}  # length -> (min_gap, parent_j)
        for (l, g, pj) in candidates:
            if l not in len_to_best or g < len_to_best[l][0]:
                len_to_best[l] = (g, pj)
        lengths_sorted = sorted(len_to_best.keys(), reverse=True)
        pareto = []
        best_gap = float('inf')
        for l in lengths_sorted:
            g, pj = len_to_best[l]
            if g < best_gap:
                pareto.append((l, g, pj))
                best_gap = g
        dp2[i] = pareto
        if pareto and pareto[0][0] > best_len2:
            best_len2 = pareto[0][0]
            best_end2 = i

    # Reconstruct
    if best_end2 == 0:
        return [p_list[0]]

    seq = [best_end2]
    cur = best_end2
    # Find the entry in dp2[cur] with length == best_len2
    target_len = best_len2
    while target_len > 1:
        found = False
        for (l, g, pj) in dp2[cur]:
            if l == target_len:
                seq.append(pj)
                cur = pj
                target_len -= 1
                found = True
                break
        if not found:
            break
    seq.reverse()
    result = [p_list[i] for i in seq]

    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if not all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
            result = [p_list[0]]

    return result

# EVOLVE-BLOCK-END
