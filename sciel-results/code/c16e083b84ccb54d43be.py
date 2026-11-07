import time
import sys

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a * b // _gcd(a, b)

PERIOD = 27720
N = 12
MAX_L = 144

# Precompute change lists for each residue m in 0..PERIOD-1
_all_changes = [None] * PERIOD
for _m in range(PERIOD):
    result = []
    for i in range(1, N + 1):
        pts = []
        k = (_m // i) + 1
        val = k * i
        while val - _m <= MAX_L:
            pts.append(val - _m)
            val += i
        result.append(pts)
    _all_changes[_m] = result

# Precompute lower bounds for each m
_lower_bounds = [0] * PERIOD
for _m in range(PERIOD):
    lo = 1
    for i in range(1, N + 1):
        d = ((_m // i) + 1) * i - _m
        if d > lo:
            lo = d
    _lower_bounds[_m] = lo

def _has_perfect_matching_bitmask(adj_bitmasks, left_size, right_size):
    """Kuhn's algorithm using bitmask adjacency. adj_bitmasks[u] is a bitmask of right nodes."""
    match_to_right = [-1] * right_size

    def dfs(u, seen_mask):
        bm = adj_bitmasks[u]
        while bm:
            v = (bm & (-bm)).bit_length() - 1
            bm &= bm - 1
            if (seen_mask >> v) & 1:
                continue
            seen_mask |= (1 << v)
            if match_to_right[v] == -1 or dfs(match_to_right[v], seen_mask):
                match_to_right[v] = u
                return True
        return False

    for u in range(left_size):
        seen_mask = 0
        if not dfs(u, seen_mask):
            return False
    return True

def _can_match(m, L):
    """Return True if (m, m+L] contains 12 distinct b_i with i | b_i."""
    adj_bitmasks = [0] * N
    changes_m = _all_changes[m]
    for i in range(N):
        bm = 0
        pts = changes_m[i]
        for Lc in pts:
            if Lc <= L:
                bm |= (1 << (Lc - 1))
            else:
                break
        adj_bitmasks[i] = bm
        if bm == 0:
            return False

    # Quick Hall's check: union of all neighbors has enough capacity
    all_union = 0
    for i in range(N):
        all_union |= adj_bitmasks[i]
    if all_union.bit_count() < N:
        return False

    return _has_perfect_matching_bitmask(adj_bitmasks, N, L)

def _f_of_m(m):
    """Least positive L such that can_match(m, L)."""
    lo = _lower_bounds[m]
    hi = MAX_L + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = 88.0  # safety margin before 90s kill

    best_m = 0
    best_L = -1

    # Check time every 1000 iterations to reduce overhead
    for m in range(PERIOD):
        if (m & 1023) == 0:
            if time.time() - start > deadline:
                break
        L = _f_of_m(m)
        if L > best_L:
            best_L = L
            best_m = m

    return int(best_m)

# EVOLVE-BLOCK-START
# Entry point: construct() -> int in [0, 1000000]
# EVOLVE-BLOCK-END
