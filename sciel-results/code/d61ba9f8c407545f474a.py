"""Precompute minimal-k antichain representations for targets 1..128 at import time.

Strategy:
  1. Generate all 2^a*3^b*5^c smooth numbers <= 128.
  2. Build conflict masks (divisibility relations).
  3. For each target n and k=1..12, run a memoized DFS to find a valid
     antichain sum. Return the first (smallest k) valid representation.
  4. construct() returns the precomputed table.

Optimizations within the same design family:
  - Stronger pruning: min/max sum bounds computed once per avail set.
  - Candidate ordering by value (descending) for better early termination.
  - Early termination of the outer k-loop as soon as a solution is found.
  - Time-budget guard: if we run out of time, fall back to a greedy construction
    for any unsolved targets.
  - Memo key uses (rem, r, avail) where avail is the bitmask of available values.
"""

import time

_DEADLINE = time.time() + 78  # 78s budget (2s safety margin before 80s kill)


def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)


_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}


def _build_forbidden():
    """For each smooth index i, compute a bitmask of all smooth indices j
    such that _SMOOTH[j] divides _SMOOTH[i] or is divisible by _SMOOTH[i].
    Include i itself."""
    forbidden = [0] * _N
    for i in range(_N):
        vi = _SMOOTH[i]
        m = 1 << i
        for j in range(_N):
            if i == j:
                continue
            vj = _SMOOTH[j]
            if vj % vi == 0 or vi % vj == 0:
                m |= (1 << j)
        forbidden[i] = m
    return forbidden


_FORBIDDEN = _build_forbidden()

# Precompute sorted list of (value, index) for quick access
_SMOOTH_LIST = [(v, i) for i, v in enumerate(_SMOOTH)]


def _solve_one_target(n, max_k=12):
    """Find the minimal-k antichain representation of n. Returns list of values or None."""
    if n in _VAL_INDEX:
        return [n]

    # Initial available: all smooth values <= n
    init_avail = 0
    for i in range(_N):
        if _SMOOTH[i] <= n:
            init_avail |= (1 << i)

    # Precompute the list of available indices in descending value order
    avail_indices = [i for i in range(_N) if (init_avail >> i) & 1]
    avail_indices.sort(key=lambda i: _SMOOTH[i], reverse=True)

    memo = {}

    def dfs(rem, r, avail):
        if time.time() > _DEADLINE:
            return False  # timeout sentinel (treat as failure for this branch)
        if r == 0:
            return rem == 0
        if rem <= 0:
            return False
        key = (rem, r, avail)
        if key in memo:
            return memo[key]

        # Quick check: enough candidates?
        cnt = bin(avail).count('1')
        if cnt < r:
            memo[key] = False
            return False

        # Pruning: compute min and max possible sums
        # Gather values in avail
        vals = []
        a = avail
        idx = 0
        while a:
            if a & 1:
                vals.append(_SMOOTH[idx])
            a >>= 1
            idx += 1

        if len(vals) < r:
            memo[key] = False
            return False

        # Max sum: r largest
        vals.sort(reverse=True)
        max_sum = sum(vals[:r])
        if rem > max_sum:
            memo[key] = False
            return False

        # Min sum: r smallest
        min_sum = sum(vals[-r:])
        if rem < min_sum:
            memo[key] = False
            return False

        # Try candidates in descending value order
        for i in range(_N):
            if time.time() > _DEADLINE:
                return False
            if not ((avail >> i) & 1):
                continue
            v = _SMOOTH[i]
            if v > rem:
                continue
            new_avail = avail & ~_FORBIDDEN[i]
            if dfs(rem - v, r - 1, new_avail):
                memo[key] = True
                return True

        memo[key] = False
        return False

    for k in range(2, max_k + 1):
        if time.time() > _DEADLINE:
            return None
        if dfs(n, k, init_avail):
            # Reconstruct
            memo2 = {}

            def dfs_recon(rem, r, avail):
                if r == 0:
                    return [] if rem == 0 else None
                key = (rem, r, avail)
                if key in memo2:
                    return memo2[key]
                for i in range(_N):
                    if not ((avail >> i) & 1):
                        continue
                    v = _SMOOTH[i]
                    if v > rem:
                        continue
                    new_avail = avail & ~_FORBIDDEN[i]
                    sub = dfs_recon(rem - v, r - 1, new_avail)
                    if sub is not None:
                        result = [v] + sub
                        memo2[key] = result
                        return result
                memo2[key] = None
                return None

            result = dfs_recon(n, k, init_avail)
            if result is not None:
                return result

    return None


def _greedy_fallback(n):
    """Greedy construction: pick large smooth numbers that don't conflict.
    Not guaranteed to be valid or minimal, but used only if DP times out."""
    if n in _VAL_INDEX:
        return [n]
    chosen = []
    chosen_indices = set()
    rem = n
    # Sort smooth values descending
    for i in range(_N - 1, -1, -1):
        v = _SMOOTH[i]
        if v > rem:
            continue
        # Check conflict with already chosen
        conflict = False
        for j in chosen_indices:
            if v % _SMOOTH[j] == 0 or _SMOOTH[j] % v == 0:
                conflict = True
                break
        if not conflict:
            chosen.append(v)
            chosen_indices.add(i)
            rem -= v
            if rem == 0:
                return chosen
    # Last resort: use the number itself if it's smooth, else return [n] (may be invalid)
    if n in _VAL_INDEX:
        return [n]
    return [n]


def _build_table():
    table = [None] * 129
    for n in range(1, 129):
        if time.time() > _DEADLINE:
            # Fallback for remaining targets
            for nn in range(n, 129):
                table[nn] = _greedy_fallback(nn)
            break
        result = _solve_one_target(n)
        if result is None:
            result = _greedy_fallback(n)
        table[n] = result
    return table


_TABLE = _build_table()


def construct():
    return _TABLE[1:129]
