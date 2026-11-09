"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

PERIOD = _compute_lcm_range(12)  # 27720
N = 12
MAX_L = 144  # upper bound on f

def _has_perfect_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
    # For each i in 1..12, find all multiples of i in (m, m+L]
    # Build adjacency: i -> list of candidate values
    # Then check perfect matching
    
    # Map each candidate value to a node index for the right side
    # Since L <= 144 and N=12, at most 144 candidates total
    # Use simple DFS matching (Hungarian-style for bipartite)
    
    # Build candidate list for each i
    adj = [[] for _ in range(N)]  # adj[i-1] = list of right-side indices
    val_to_idx = {}
    idx = 0
    for i in range(1, N + 1):
        # smallest multiple of i > m
        first = ((m // i) + 1) * i
        v = first
        while v <= m + L:
            if v not in val_to_idx:
                val_to_idx[v] = idx
                idx += 1
            adj[i - 1].append(val_to_idx[v])
            v += i
    
    if idx == 0:
        return False
    
    # Simple DFS-based bipartite matching
    match_r = [-1] * idx  # match_r[j] = left node matched to right node j
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    result = 0
    for u in range(N):
        if not adj[u]:
            return False
        visited = [False] * idx
        if dfs(u, visited):
            result += 1
    
    return result == N

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has valid assignment."""
    # Binary search on L
    lo, hi = 1, MAX_L
    best = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best if best != -1 else 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Precompute f for all residues 0..27719
    # But we need m in [0, 1000000], so m mod 27720 determines f for large m
    # For small m, we need to check directly
    
    # Strategy: check all residues, find the max f, return the smallest m with that f
    # Actually, f(m) depends on m mod 27720 for the divisibility pattern,
    # but the actual values in (m, m+L] depend on m itself.
    # However, if m is large enough (m >= 27720), then m mod 27720 determines
    # which residues appear, and since L <= 144 << 27720, the pattern is the same.
    # So for m >= 27720, f(m) = f(m mod 27720) ... wait, not exactly.
    # If m = k*27720 + r, then (m, m+L] = (k*27720+r, k*27720+r+L]
    # A number x in this range is divisible by i iff x mod i == 0,
    # which depends on x mod lcm(1..12) = 27720.
    # So x mod 27720 ranges from (r+1) mod 27720 to (r+L) mod 27720 (no wrap since L<27720).
    # So f(m) depends only on r = m mod 27720, for any m.
    # Great! So I just need to compute f for r in 0..27719 and pick the best.
    
    best_f = 0
    best_m = 0
    
    # Check all residues
    for r in range(PERIOD):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    # best_m is in [0, 27719], which is within [0, 1000000]
    # Return as Python int
    return int(best_m)

# EVOLVE-BLOCK-END
