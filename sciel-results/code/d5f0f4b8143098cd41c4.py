import time
import sys
from itertools import permutations
import numpy as np

def _classic_construct(n):
    """Classic recursive superpermutation construction, optimal for small n."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _build_word_from_perm_order(order, n):
    """
    Given an ordering of permutations (list of tuples), build the superpermutation word.
    """
    if not order:
        return ""
    word = list(order[0])
    for i in range(1, len(order)):
        u = tuple(order[i - 1])
        v = order[i]
        # Find max overlap
        max_ov = n - 1
        ov = 0
        for l in range(max_ov, 0, -1):
            if u[n-l:] == v[:l]:
                ov = l
                break
        word.extend(list(v[ov:]))
    return "".join(map(str, word))

def _get_cycles(perms, n):
    """
    Find the disjoint cycles formed by the 'cost 1' edges.
    Edge P -> Q if P[1:] == Q[:n-1].
    """
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    N = len(perms)
    
    # Find successor for each perm
    # Successor of P is the unique Q such that Q[:n-1] == P[1:]
    # Q = (P[1], ..., P[n-1], missing_elem)
    succ = [None] * N
    for i, p in enumerate(perms):
        prefix = p[1:]
        # Find missing element
        s = set(range(1, n+1))
        used = set(prefix)
        missing = (s - used).pop()
        q = prefix + (missing,)
        j = perm_to_idx[q]
        succ[i] = j
        
    # Trace cycles
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = succ[curr]
            cycles.append(cycle)
            
    return cycles

def _construct_via_cycles(n, deadline):
    """
    Construct superpermutation by optimizing the order of cycles.
    """
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    if n <= 4:
        return _classic_construct(n)
        
    cycles = _get_cycles(perms, n)
    K = len(cycles)
    
    # For each pair of cycles (i, j), compute the 7x7 cost matrix
    # cost[i][j][si][sj] = cost to go from cycle i (entering at offset si) 
    # to cycle j (entering at offset sj)
    # Exit offset of i is (si + n - 1) % n
    # Entry offset of j is sj
    # Cost is distance from perms[cycles[i][(si+n-1)%n]] to perms[cycles[j][sj]]
    
    # We only need the minimum cost for the TSP heuristic on cycles
    # min_cost[i][j] = min over si, sj of cost[i][j][si][sj]
    
    # Precompute transition costs between all pairs of permutations?
    # N=5040. N^2 = 25M.
    # We can compute this on the fly for the K^2 pairs of cycles.
    # K=720. K^2 = 518k.
    # Each pair requires 49 distance checks.
    # Total 25M distance checks.
    # Distance check is O(n).
    
    # To speed up, we can precompute a dictionary of costs for specific pairs?
    # Or just compute min_cost[i][j] directly.
    
    min_cost = np.zeros((K, K), dtype=np.int8)
    
    # Helper to compute distance between two perms
    def dist(p1, p2):
        # p1, p2 are tuples
        # Find max overlap
        # p1[n-l:] == p2[:l]
        # l from n-1 down to 0
        for l in range(n-1, -1, -1):
            if l == 0 or p1[n-l:] == p2[:l]:
                return n - l
        return n # Should not happen
        
    for i in range(K):
        ci = cycles[i]
        for j in range(K):
            if i == j: continue
            cj = cycles[j]
            min_c = n # Max cost is n
            for si in range(n):
                end_i = ci[(si + n - 1) % n]
                p_end = perms[end_i]
                for sj in range(n):
                    start_j = cj[sj]
                    p_start = perms[start_j]
                    c = dist(p_end, p_start)
                    if c < min_c:
                        min_c = c
            min_cost[i][j] = min_c
            
    if time.time() > deadline - 60:
        return _classic_construct(n)
        
    # TSP heuristic on K cycles
    # Greedy NN
    visited = [False] * K
    visited[0] = True
    path = [0]
    curr = 0
    
    for _ in range(K - 1):
        # Find nearest unvisited
        best_j = -1
        best_d = float('inf')
        for j in range(K):
            if not visited[j]:
                d = min_cost[curr][j]
                if d < best_d:
                    best_d = d
                    best_j = j
        path.append(best_j)
        visited[best_j] = True
        curr = best_j
        
    # 2-Opt improvement on the cycle path
    improved = True
    max_iters = 50
    iter_count = 0
    while improved and iter_count < max_iters and time.time() < deadline - 30:
        improved = False
        iter_count += 1
        for i in range(1, K - 1):
            for j in range(i + 1, K):
                # Reverse path[i:j+1]
                # Check cost change
                # Edges: (i-1, i), (j, j+1)
                # New: (i-1, j), (i, j+1)
                
                old_cost = 0
                new_cost = 0
                
                # i-1 is always valid since i>=1
                old_cost += min_cost[path[i-1]][path[i]]
                if j < K - 1:
                    old_cost += min_cost[path[j]][path[j+1]]
                
                new_cost += min_cost[path[i-1]][path[j]]
                if j < K - 1:
                    new_cost += min_cost[path[i]][path[j+1]]
                    
                if new_cost < old_cost:
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
                
    # Now we have an order of cycles.
    # We need to determine the optimal offset for each cycle in this order.
    # DP: dp[i][s] = min cost to reach cycle i with entry offset s.
    # dp[0][s] = 0 (cost to start is 0, we just pick the start)
    # Actually, the cost is sum of link costs.
    # Link from i-1 (exit s_{i-1}) to i (entry s_i) costs C_{i-1, i}[s_{i-1}, s_i].
    
    # We need the full cost matrix for the specific pairs in the path.
    # cost_link[i][si][sj] for path[i-1] -> path[i]
    
    # Let's store the best offsets
    best_prev_offset = [[-1]*n for _ in range(K)]
    dp = [[float('inf')]*n for _ in range(K)]
    
    for s in range(n):
        dp[0][s] = 0
        
    for idx in range(1, K):
        prev_cycle_idx = path[idx-1]
        curr_cycle_idx = path[idx]
        ci = cycles[prev_cycle_idx]
        cj = cycles[curr_cycle_idx]
        
        for sj in range(n):
            min_val = float('inf')
            best_si = -1
            for si in range(n):
                if dp[idx-1][si] == float('inf'): continue
                end_i = ci[(si + n - 1) % n]
                start_j = cj[sj]
                c = dist(perms[end_i], perms[start_j])
                val = dp[idx-1][si] + c
                if val < min_val:
                    min_val = val
                    best_si = si
            dp[idx][sj] = min_val
            best_prev_offset[idx][sj] = best_si
            
    # Backtrack to find best offsets
    best_final_offset = min(range(n), key=lambda s: dp[K-1][s])
    offsets = [0]*K
    offsets[K-1] = best_final_offset
    
    for idx in range(K-1, 0, -1):
        prev_si = best_prev_offset[idx][offsets[idx]]
        offsets[idx-1] = prev_si
        
    # Construct the final permutation order
    final_order = []
    for idx in range(K):
        c = cycles[path[idx]]
        s = offsets[idx]
        # The cycle is c[0], c[1], ..., c[n-1]
        # We enter at s, so we visit c[s], c[s+1], ..., c[s+n-1] (mod n)
        for k in range(n):
            final_order.append(perms[c[(s+k) % n]])
            
    word = _build_word_from_perm_order(final_order, n)
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    if n <= 4:
        return _classic_construct(n)
    
    # Try the cycle-based optimization
    try:
        word = _construct_via_cycles(n, deadline)
        return word
    except Exception:
        pass
        
    # Fallback to classic
    return _classic_construct(n)
