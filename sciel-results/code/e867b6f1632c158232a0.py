import time
import math

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1..12)
    MAX_L = 144  # Upper bound for L
    
    # Precompute candidates for each residue r and each i
    # all_cands[r][i] is a list of j in [0, MAX_L) such that i divides (r + 1 + j)
    # Note: r is 0-indexed residue of m. Interval is (m, m+L], so values are m+1 ... m+L.
    # If m % PERIOD == r, then m+1+k has residue (r+1+k) % PERIOD.
    # We need i | (m+1+k).
    
    all_cands = [None] * PERIOD
    for r in range(PERIOD):
        cands = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            # Find j in [0, MAX_L) such that i | (r + 1 + j)
            # r + 1 + j ≡ 0 (mod i)
            # j ≡ -r - 1 (mod i)
            target_res = (-r - 1) % i
            # j can be target_res, target_res + i, target_res + 2i, ...
            j = target_res
            while j < MAX_L:
                cands[i].append(j)
                j += i
        all_cands[r] = cands

    def has_matching(r, L):
        # r is the residue, interval is (r, r+L]
        # Left nodes: 1..N
        # Right nodes: indices 0..L-1 corresponding to values r+1 .. r+L
        
        cands_r = all_cands[r]
        
        # Filter candidates to those < L
        # And sort left nodes by number of valid candidates to speed up Kuhn's
        left_nodes = []
        for i in range(1, N + 1):
            valid_cands = [j for j in cands_r[i] if j < L]
            if not valid_cands:
                return False
            left_nodes.append((len(valid_cands), i, valid_cands))
        
        # Sort by fewest candidates first
        left_nodes.sort(key=lambda x: x[0])
        
        match_to = [-1] * L
        
        # Kuhn's algorithm
        # We use a visited array for the right nodes in each DFS call
        # To optimize, we can use a timestamp array instead of resetting visited each time?
        # L is small (<145), so resetting is cheap.
        
        def dfs(i, visited):
            for j in left_nodes_map[i]:
                if not visited[j]:
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i
                        return True
            return False
        
        # We need to map i to its valid_cands quickly inside dfs
        # Since i is 1..12, we can use a list of size 13
        left_nodes_map = [None] * (N + 1)
        for _, i, vc in left_nodes:
            left_nodes_map[i] = vc
            
        for _, i, _ in left_nodes:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_m = 0
    max_L_found = 0
    
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    # We can iterate r in any order.
    # To potentially find a high L earlier, maybe randomize?
    # But exhaustive is required for certainty unless we find 144.
    # The max possible L is bounded. The problem states "at most 144".
    # If we find 144, we can stop.
    
    for r in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Binary search for min L
        # L must be at least N=12.
        # If has_matching(r, MAX_L) is false, skip (should not happen if 144 is max possible and valid)
        if not has_matching(r, MAX_L):
            continue
            
        lo, hi = 12, MAX_L
        # Find smallest L such that has_matching(r, L) is True
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo > max_L_found:
            max_L_found = lo
            best_m = r
            if max_L_found == MAX_L:
                return best_m

    return best_m
