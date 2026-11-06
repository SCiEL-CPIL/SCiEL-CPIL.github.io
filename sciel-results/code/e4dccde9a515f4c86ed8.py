"""Equal-gcd-triple-free subsets of [1,128].

Branch-and-bound with conflict-degree ordering and compatibility upper bound.
"""
import time
import math
import random
from itertools import combinations

_DEADLINE = 88.0
_START = None
_N = 128
_CAND = list(range(1, _N + 1))
_PRE = None
_BUCKETS = None
_CONFLICT_DEG = None
_ORDERED = None
_BEST = [0, 0]  # [size, mask]


def _ensure_pre():
    global _PRE, _BUCKETS, _CONFLICT_DEG, _ORDERED
    if _PRE is not None:
        return
    PRE = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            PRE[i][j] = PRE[j][i] = math.gcd(i, j)
    _PRE = PRE

    BUCKETS = [None] * (_N + 1)
    for i in range(1, _N + 1):
        d = {}
        row = PRE[i]
        for j in range(1, _N + 1):
            g = row[j]
            d.setdefault(g, []).append(j)
        BUCKETS[i] = d
    _BUCKETS = BUCKETS

    # Precompute conflict degree: for each x, count triples (x,a,b) with x<a<b
    # such that gcd(x,a)=gcd(x,b)=gcd(a,b)
    conflict_deg = [0] * (_N + 1)
    for x in range(1, _N + 1):
        for a in range(x + 1, _N + 1):
            gxa = PRE[x][a]
            for b in range(a + 1, _N + 1):
                if PRE[x][b] == gxa and PRE[a][b] == gxa:
                    conflict_deg[x] += 1
    _CONFLICT_DEG = conflict_deg

    # Order by conflict degree (most constrained first)
    ordered = sorted(range(1, _N + 1), key=lambda x: -conflict_deg[x])
    _ORDERED = ordered


def _can_add_to_set(S_set, x):
    """Check if x can be added to S_set (a set of ints) without creating violation."""
    P = _PRE
    for a in S_set:
        gxa = P[x][a]
        for b in S_set:
            if a >= b:
                continue
            if P[x][b] == gxa and P[a][b] == gxa:
                return False
    return True


def _compatible_count(S_set, remaining):
    """Upper bound: count how many elements from 'remaining' can be added to S_set.
    This is a loose bound: we greedily add compatible elements."""
    P = _PRE
    S = set(S_set)
    added = 0
    for x in remaining:
        if x in S:
            continue
        # Check if x is compatible with all in S
        ok = True
        for a in S:
            gxa = P[x][a]
            for b in S:
                if a >= b:
                    continue
                if P[x][b] == gxa and P[a][b] == gxa:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.add(x)
            added += 1
    return added


def _branch_and_bound(depth, S_set, mask, remaining, deadline):
    """Recursive branch-and-bound. Returns (best_size, best_mask)."""
    P = _PRE
    current_size = len(S_set)
    
    # Update best if current set is better
    if current_size > _BEST[0]:
        _BEST[0] = current_size
        _BEST[1] = mask
    
    # If no remaining elements, return
    if not remaining:
        return _BEST[0], _BEST[1]
    
    # Upper bound: current_size + len(remaining)
    # If this can't beat best, prune
    if current_size + len(remaining) <= _BEST[0]:
        return _BEST[0], _BEST[1]
    
    # Time check
    if time.time() > deadline:
        return _BEST[0], _BEST[1]
    
    # Pick next element (first in remaining, which is ordered by conflict degree)
    x = remaining[0]
    rest = remaining[1:]
    
    # Branch 1: Include x
    if _can_add_to_set(S_set, x):
        new_S = S_set | {x}
        new_mask = mask | (1 << (x - 1))
        _branch_and_bound(depth + 1, new_S, new_mask, rest, deadline)
    
    # Branch 2: Exclude x
    _branch_and_bound(depth + 1, S_set, mask, rest, deadline)
    
    return _BEST[0], _BEST[1]


def _construct():
    global _START, _BEST
    _ensure_pre()
    _START = time.time()
    _BEST = [0, 0]
    
    deadline = _DEADLINE - 2
    
    # Start with empty set, branch-and-bound over ordered elements
    S_set = set()
    mask = 0
    remaining = _ORDERED[:]
    
    _branch_and_bound(0, S_set, mask, remaining, deadline)
    
    best_size, best_mask = _BEST
    best_list = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    
    # If we got a valid set, try to improve with local add
    if best_list and _violations_mask(best_mask) == 0:
        S, new_mask = _local_add(best_mask, best_list)
        if len(S) > best_size:
            best_size = len(S)
            best_mask = new_mask
            best_list = S
    
    if not best_list:
        best_list = [2]
    return best_list


def _violations_mask(mask):
    """Number of equal-gcd triples in the set encoded by bit-mask."""
    P = _PRE
    cnt = 0
    s = [i for i in range(1, _N + 1) if (mask >> (i - 1)) & 1]
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g and P[b][c] == g:
                    cnt += 1
    return cnt


def _local_add(mask, S):
    """Add any excluded element that fits, repeatedly."""
    P = _PRE
    changed = True
    S = list(S)
    S_set = set(S)
    while changed:
        changed = False
        for x in _CAND:
            if x in S_set:
                continue
            if _can_add_to_set(S_set, x):
                S_set.add(x)
                S.append(x)
                mask |= 1 << (x - 1)
                changed = True
    return S, mask


def construct():
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        res = [2]
    return res
