import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MOD = 101
    N_DENOMS = 10
    
    # Precompute modular inverses
    invs = [0] * (N_DENOMS + 1)
    for x in range(1, N_DENOMS + 1):
        invs[x] = pow(x, -1, MOD)
    
    # Generate all non-empty subsets of {1, ..., 10}
    # For each subset, compute: sum of inverses mod 101, size, actual denominators
    subset_info = []  # list of (residue, size, denominators_list, mask)
    
    for mask in range(1, 1 << N_DENOMS):
        s = 0
        size = 0
        denoms = []
        for x in range(1, N_DENOMS + 1):
            if mask & (1 << (x - 1)):
                denoms.append(x)
                s = (s + invs[x]) % MOD
                size += 1
        subset_info.append((s, size, denoms, mask))
    
    n_subsets = len(subset_info)
    n_residues = MOD
    
    # For each residue, store which subsets achieve it
    residue_to_subsets = [[] for _ in range(n_residues)]
    for i, (res, sz, denoms, mask) in enumerate(subset_info):
        residue_to_subsets[res].append(i)
    
    # Check that all residues are covered
    for r in range(n_residues):
        if not residue_to_subsets[r]:
            # Should not happen, but handle gracefully
            pass
    
    # Hopcroft-Karp algorithm for bipartite matching
    # Left side: residues (0..100), Right side: subsets (0..1022)
    # Edge exists if subset's sum equals residue
    # We want to find min k such that perfect matching exists using only subsets of size <= k
    
    def hopcroft_karp(max_k):
        """
        Find perfect matching using only subsets of size <= max_k.
        Returns (success, match_l) where match_l[r] = subset index assigned to residue r.
        """
        # Build adjacency: for each residue, list of valid subsets (size <= max_k)
        adj = [[] for _ in range(n_residues)]
        for r in range(n_residues):
            for si in residue_to_subsets[r]:
                if subset_info[si][1] <= max_k:
                    adj[r].append(si)
        
        # If any residue has no valid subsets, matching impossible
        for r in range(n_residues):
            if not adj[r]:
                return False, None
        
        match_l = [-1] * n_residues   # match_l[r] = subset matched to residue r
        match_r = [-1] * n_subsets    # match_r[s] = residue matched to subset s
        dist = [0] * n_residues
        
        def bfs():
            queue = deque()
            found = False
            for u in range(n_residues):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = match_r[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == float('inf'):
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        # Increase recursion limit for DFS depth
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        
        matching_size = 0
        try:
            while bfs():
                for u in range(n_residues):
                    if match_l[u] == -1:
                        if dfs(u):
                            matching_size += 1
                if time.time() - start_time > DEADLINE - 15:
                    sys.setrecursionlimit(old_limit)
                    return False, None
            sys.setrecursionlimit(old_limit)
        except Exception:
            sys.setrecursionlimit(old_limit)
            return False, None
        
        if matching_size == n_residues:
            return True, match_l
        return False, match_l
    
    # Try increasing max_k until we find a perfect matching
    best_result = None
    for max_k in range(1, N_DENOMS + 1):
        if time.time() - start_time > DEADLINE - 15:
            break
        success, match_l = hopcroft_karp(max_k)
        if success:
            # Build result
            result = []
            for r in range(n_residues):
                subset_idx = match_l[r]
                result.append(subset_info[subset_idx][2])
            return result
    
    # Fallback: greedy - assign minimum size subset to each residue
    if time.time() - start_time > DEADLINE - 10:
        # Just return something valid
        result = []
        for r in range(n_residues):
            if residue_to_subsets[r]:
                # Pick the smallest subset
                best_si = min(residue_to_subsets[r], key=lambda i: subset_info[i][1])
                result.append(subset_info[best_si][2])
            else:
                result.append([1])  # fallback, should not happen
        return result
    
    # More careful fallback: try to find assignment manually
    # Use a simple greedy with backtracking or just minimum per residue
    min_len = [N_DENOMS + 1] * n_residues
    best_subset = [None] * n_residues
    
    for i, (res, sz, denoms, mask) in enumerate(subset_info):
        if sz < min_len[res]:
            min_len[res] = sz
            best_subset[res] = denoms
    
    result = []
    for r in range(n_residues):
        if best_subset[r] is None:
            # Find any subset that sums to r
            for i, (res, sz, denoms, mask) in enumerate(subset_info):
                if res == r:
                    best_subset[r] = denoms
                    break
        if best_subset[r] is None:
            best_subset[r] = [1]  # last resort
        result.append(best_subset[r])
    
    return result
