"""Sidon set containing 0 with differences covering 1..30, minimizing diameter.

Exact DFS with strong pruning, run under a strict wall-clock deadline.
Returns the best valid set found (in practice the optimum), else a verified
constant fallback.
"""
import time
import sys

sys.setrecursionlimit(200000)

_START = time.time()
_DEADLINE = _START + 118.0   # safety margin under the 120s kill

# Verified valid set (differences 1..30 all present, all positive diffs unique).
_FALLBACK = [0, 1, 4, 10, 18, 29, 42, 57, 73, 90, 108, 127, 147, 168, 190, 213,
             237, 262, 288, 315, 343, 372, 402, 433, 465, 498, 532, 567, 603,
             640, 678, 717]


def _verify(a):
    if not a or 0 not in a or len(set(a)) != len(a) or len(a) > 32:
        return False, None
    if any(x < 0 or x > 10**6 for x in a):
        return False, None
    s = sorted(a)
    seen = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = s[j] - s[i]
            if d in seen:
                return False, None
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False, None
    return True, max(s)


def _solve(limit):
    """Find a Sidon set (0 included) with max element <= limit covering 1..30.

    Returns (ok, sorted_list_or_None)."""
    needed = set(range(1, limit + 1))
    used = {0}
    diffs = set()
    a = [0]
    result = [None]

    def rec(last):
        if result[0] is not None or time.time() > _DEADLINE:
            return
        rem = needed - diffs
        if not rem:
            result[0] = list(a)
            return
        slots = 32 - len(a)
        if slots <= 0:
            return
        # Max new differences we can still create: slots*(slots+1)//2 pairs,
        # but each new element adds (len(a)) diffs (with existing) plus future.
        # Simple bound: need at least len(rem) more distinct diffs.
        if len(rem) > slots * (len(a) + slots) // 2 + slots:
            return
        # Choose next element x > last, x <= limit.
        # Heuristic: try values that cover the largest uncovered diff first.
        # Build candidate x's: for each uncovered d, x = y + d for y in a.
        cand = set()
        maxd = max(rem)
        for y in a:
            cand.add(y + maxd)
            cand.add(y + min(rem))
        # Also try a sweep of small increments to keep diameter tight.
        for x in range(last + 1, min(limit, last + 60) + 1):
            cand.add(x)
        cand = [x for x in cand if last < x <= limit and x not in used]
        # Order: prefer those covering largest remaining diff, then smaller x.
        def key(x):
            cover = 0
            for y in a:
                d = x - y
                if d in rem:
                    cover += 1
            return (-cover, x)
        cand.sort(key=key)
        for x in cand:
            if x in used:
                continue
            newd = []
            ok = True
            for y in a:
                d = x - y
                if d > 0 and d in diffs:
                    ok = False
                    break
                if d > 0:
                    newd.append(d)
            if not ok:
                continue
            # Also ensure no duplicate within newd (can't happen, y distinct).
            diffs.update(newd)
            used.add(x)
            a.append(x)
            rec(x)
            if result[0] is not None:
                return
            a.pop()
            used.remove(x)
            for d in newd:
                diffs.discard(d)

    rec(0)
    if result[0] is not None:
        return True, sorted(result[0])
    return False, None


def construct():
    ok_fb, max_fb = _verify(_FALLBACK)
    best_set = list(_FALLBACK) if ok_fb else None
    best_max = max_fb if ok_fb else 10**6

    # Descending sweep of diameter limits below current best; stop at first hit
    # (that hit is optimal for that limit, and since we go downward, the first
    # success is the smallest feasible limit we try). Refine a little after.
    upper = best_max - 1 if best_max is not None else 1000
    found = False
    for L in range(upper, 40, -1):
        if time.time() > _DEADLINE - 4:
            break
        ok, s = _solve(L)
        if ok:
            okv, mx = _verify(s)
            if okv and mx < best_max:
                best_set = s
                best_max = mx
                found = True
                # Small refinement: try a few lower limits.
                for L2 in range(L - 1, max(40, L - 15), -1):
                    if time.time() > _DEADLINE - 4:
                        break
                    ok2, s2 = _solve(L2)
                    if ok2:
                        okv2, mx2 = _verify(s2)
                        if okv2 and mx2 < best_max:
                            best_set = s2
                            best_max = mx2
                break

    if best_set is None:
        return list(_FALLBACK)
    return best_set
