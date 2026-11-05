import time
import math
from functools import lru_cache

_N = 5040
_T = 4979

def _all_divisors(m):
    out = []
    for i in range(1, int(m ** 0.5) + 1):
        if m % i == 0:
            out.append(i)
            if i != m // i:
                out.append(m // i)
    return out

def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before 90 s kill

    n = 5040
    target = 4979

    divs = _all_divisors(n)
    # divisors >= target can never be used in a positive-sum representation
    pool = [d for d in divs if d <= target]
    # largest first -> greedy upper bound is tight and pruning is aggressive
    pool.sort(reverse=True)
    m = len(pool)

    # suffix sums for a feasibility upper bound
    suff = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suff[i] = suff[i + 1] + pool[i]

    # quick greedy upper bound
    rem = target
    ub = 0
    for d in pool:
        if d <= rem:
            rem -= d
            ub += 1
            if rem == 0:
                break
    if rem != 0:
        # greedy with largest-first is not guaranteed; fall back to any bound
        ub = m

    best = [ub]
    bestset = [set()]

    # Precompute a memoized feasibility table:
    # can(i, rem, k) = True iff we can pick exactly k distinct divisors from pool[i:] summing to rem
    # We'll use this to prune the DFS aggressively.
    # Since m can be up to ~60 and rem up to 4979, we use a set-based approach.
    # For each starting index i and remaining sum rem, track the set of achievable counts.
    # Actually, let's do a different approach: for each i, compute which (rem, count) pairs are achievable.
    
    # Better: use meet-in-the-middle or just memoize the DFS feasibility check.
    # Let's precompute for each position i, the set of (sum, min_count) that are achievable.
    # Actually, simplest: memoize a function feasible_exact(i, rem, k) using lru_cache.
    
    # But k varies, so let's just do: for each i and rem, what's the minimum number of elements needed?
    # min_count[i][rem] = minimum number of distinct elements from pool[i:] that sum to rem, or inf if impossible.
    
    # This is a DP: min_count[i][rem] = min over j>=i where pool[j]<=rem of (1 + min_count[j+1][rem-pool[j]])
    # But rem can be up to 4979 and i up to ~60, so this is 60*4979 ~ 300k states, each with up to 60 transitions.
    # That's ~18M operations, which is fine.
    
    INF = float('inf')
    max_rem = target
    # min_count[i][rem] for i in 0..m, rem in 0..max_rem
    # Initialize: min_count[m][0] = 0, min_count[m][rem>0] = inf
    min_count = [[INF] * (max_rem + 1) for _ in range(m + 1)]
    min_count[m][0] = 0
    
    # Fill from i=m-1 down to 0
    for i in range(m - 1, -1, -1):
        d = pool[i]
        for rem in range(max_rem + 1):
            # Option 1: skip pool[i]
            val = min_count[i + 1][rem]
            # Option 2: take pool[i]
            if rem >= d:
                take_val = 1 + min_count[i + 1][rem - d]
                if take_val < val:
                    val = take_val
            min_count[i][rem] = val
    
    # Now min_count[i][rem] gives the minimum number of elements from pool[i:] that sum to rem.
    # If min_count[i][rem] == INF, it's impossible.
    
    # Now do iterative deepening DFS with strong pruning.
    chosen = []
    
    def dfs(i, rem, terms_left):
        if time.time() > DEADLINE:
            return True  # signal abort
        if rem == 0:
            if terms_left >= 0:
                cur = len(chosen)
                if cur < best[0]:
                    best[0] = cur
                    bestset[0] = set(chosen[:])
            return False
        if i >= m or rem < 0 or terms_left < 0:
            return False
        # Use the precomputed min_count for strong pruning
        if min_count[i][rem] == INF:
            return False
        if min_count[i][rem] > terms_left:
            return False
        # Also, we can't use more elements than available
        if terms_left > m - i:
            return False
        # Upper bound: even taking the largest terms_left elements, can we reach rem?
        if terms_left > 0 and rem > terms_left * pool[i]:
            return False
        
        for j in range(i, m):
            d = pool[j]
            if d > rem:
                continue
            # Prune: if even taking pool[j] and the best (terms_left-1) from pool[j+1:] can't work
            if min_count[j + 1][rem - d] == INF or min_count[j + 1][rem - d] > terms_left - 1:
                continue
            chosen.append(d)
            if dfs(j + 1, rem - d, terms_left - 1):
                chosen.pop()
                return True
            chosen.pop()
        return False

    for depth in range(1, ub + 1):
        if dfs(0, target, depth):
            break  # aborted by deadline
        if best[0] <= depth:
            # found a solution at this depth; deeper is worse, stop
            break

    if bestset[0]:
        return list(bestset[0])

    # fallback: return best effort greedy if any (should not happen)
    rem = target
    out = []
    for d in pool:
        if d <= rem:
            out.append(d)
            rem -= d
            if rem == 0:
                break
    return out
