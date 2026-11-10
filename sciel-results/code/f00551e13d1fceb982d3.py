"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
MAX_M = 1000000
DEADLINE = 85.0  # safety margin before 90s kill

def _has_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_N with i | b_i."""
    # For each i in 1..N, candidate b_i values in (m, m+L] divisible by i
    # Build bipartite graph and check perfect matching via DFS
    n = N
    # candidates[i] = list of values in (m, m+L] divisible by i
    candidates = [None] * (n + 1)
    for i in range(1, n + 1):
        # smallest multiple of i > m
        first = (m // i + 1) * i
        if first > m + L:
            return False
        # build list
        cands = list(range(first, m + L + 1, i))
        if len(cands) == 0:
            return False
        candidates[i] = cands

    # DFS-based bipartite matching (left = indices 1..N, right = values)
    # Use Hopcroft-Karp for efficiency
    # Actually for N=12, simple DFS matching suffices
    match_to_val = {}  # value -> index assigned
    result = [None] * (n + 1)  # index -> value

    def dfs(i, visited):
        for v in candidates[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_to_val or dfs(match_to_val[v], visited):
                match_to_val[v] = i
                result[i] = v
                return True
        return False

    matched = 0
    for i in range(1, n + 1):
        visited = set()
        if dfs(i, visited):
            matched += 1
        else:
            return False
    return True

def _compute_f(m):
    """Compute f(N, m): least positive L such that (m, m+L] has valid matching."""
    # Binary search on L. Lower bound: 1. Upper bound: lcm(1..12) = 27720
    lo = 1
    hi = 27720  # lcm(1..12)
    # Quick check: is L=1 possible?
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _search_optimal_m():
    """Search for m that maximizes f(12, m)."""
    start_time = time.time()
    best_m = 12
    best_f = _compute_f(12)

    # Strategy: sample m values, focusing on values that might create
    # "gaps" in divisibility coverage.
    # Key insight: f(12,m) is large when there's a long interval after m
    # where some small divisor i has few multiples available.
    # Try m values near multiples of small numbers to create scarcity.

    # First pass: coarse grid
    step = 1000
    candidates = list(range(0, MAX_M + 1, step))
    # Also add some targeted candidates
    for k in range(1, 100):
        candidates.append(k * 27720)  # multiples of lcm
        candidates.append(k * 27720 - 1)
        candidates.append(k * 27720 + 1)

    # Deduplicate and sort
    candidates = sorted(set(c for c in candidates if 0 <= c <= MAX_M))

    # Evaluate candidates within time budget
    for m in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    # Second pass: local search around best
    if time.time() - start_time < DEADLINE - 10:
        for delta in range(-50, 51):
            m = best_m + delta
            if 0 <= m <= MAX_M and time.time() - start_time > DEADLINE - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m

    # Third pass: random sampling if time remains
    if time.time() - start_time < DEADLINE - 5:
        import random
        random.seed(42)
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            m = random.randint(0, MAX_M)
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m

    return best_m

# Perform search at import time
_OPTIMAL_M = _search_optimal_m()

def construct():
    return int(_OPTIMAL_M)
# EVOLVE-BLOCK-END
