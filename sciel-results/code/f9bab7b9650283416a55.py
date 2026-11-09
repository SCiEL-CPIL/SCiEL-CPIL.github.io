"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
LCM_12 = 27720  # lcm(1..12)

def _has_perfect_matching(interval_start, interval_len):
    """Check if (interval_start, interval_start + interval_len] contains distinct b_i with i | b_i."""
    # For each index i (1..12), find which positions in the interval are divisible by i
    # We need a perfect matching: assign each i to a distinct position divisible by i.
    # Use bitmask DP or Hopcroft-Karp. For small n=12, bitmask DP is fine.
    
    # positions are interval_start+1, ..., interval_start+interval_len
    # For each i, the set of positions divisible by i.
    
    # We'll use a simple recursive backtracking with memoization on the mask of used positions.
    # But interval_len can be up to ~144, so 2^144 is too big.
    
    # Better: use the fact that for each i, the positions divisible by i are at specific offsets.
    # Actually, let's use a bipartite matching approach with the Hungarian algorithm or simple DFS.
    
    # Since n=12 is small, we can use a DFS-based matching.
    
    # Build adjacency: for each i (1..12), which positions (0-indexed within interval) are divisible by i.
    adj = []
    for i in range(1, N + 1):
        positions = []
        for pos in range(1, interval_len + 1):
            if (interval_start + pos) % i == 0:
                positions.append(pos - 1)  # 0-indexed
        adj.append(positions)
    
    # DFS-based bipartite matching
    match_to_pos = [-1] * interval_len  # which i is matched to position j
    
    def dfs(i, visited):
        for pos in adj[i]:
            if visited[pos]:
                continue
            visited[pos] = True
            if match_to_pos[pos] == -1 or dfs(match_to_pos[pos], visited):
                match_to_pos[pos] = i
                return True
        return False
    
    result = 0
    for i in range(N):
        visited = [False] * interval_len
        if dfs(i, visited):
            result += 1
    
    return result == N

def _f(interval_start, interval_len):
    """Check if f(12, m) >= interval_len, i.e., (m, m+L] has the matching."""
    return _has_perfect_matching(interval_start, interval_len)

def _compute_f_for_m(m):
    """Compute f(12, m): the least positive L such that (m, m+L] contains distinct b_i with i|b_i."""
    # Binary search on L
    lo, hi = 1, 200  # f is at most 144 per task description, but let's be safe
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _search_optimal_m(max_m=1000000, deadline=None):
    """Search for m that maximizes f(12, m)."""
    best_m = 0
    best_f = 0
    start_time = time.time()
    
    # First, do a coarse search over residues mod LCM_12
    # Since the divisibility pattern repeats every LCM_12, f(12, m) depends only on m mod LCM_12
    # Wait, that's not quite right. f(12, m) is the LEAST L such that (m, m+L] works.
    # The pattern of which numbers are divisible by which i repeats every LCM_12.
    # So f(12, m) = f(12, m + k*LCM_12) for any k.
    # Therefore, we only need to search m in [0, LCM_12 - 1] = [0, 27719].
    
    # Let's verify this reasoning: if m' = m + k*LCM_12, then for any L,
    # (m, m+L] and (m', m'+L] have the same divisibility pattern shifted by k*LCM_12.
    # Since LCM_12 divides all numbers 1..12, the divisibility of (m+pos) by i depends on
    # (m+pos) mod i, which has period LCM_12. So yes, f(12, m) is periodic with period LCM_12.
    
    # So we search m in [0, 27719].
    deadline_check_interval = 1000
    
    for m in range(0, LCM_12):
        if deadline is not None:
            if time.time() > deadline:
                break
        f_val = _compute_f_for_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        if m % deadline_check_interval == 0 and deadline is not None:
            if time.time() > deadline:
                break
    
    return best_m, best_f

def construct():
    """Return the optimal m in [0, 1000000] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 85  # 5-second safety margin
    
    best_m, best_f = _search_optimal_m(deadline=deadline)
    
    # Verify the result
    if best_f == 0:
        # Fallback: return a reasonable default
        return 12
    
    # The optimal m in [0, LCM_12) is best_m.
    # Since f is periodic with period LCM_12, any m = best_m + k*LCM_12 works.
    # We need m in [0, 1000000]. best_m is already in [0, 27719] which is within range.
    return int(best_m)

# EVOLVE-BLOCK-END
