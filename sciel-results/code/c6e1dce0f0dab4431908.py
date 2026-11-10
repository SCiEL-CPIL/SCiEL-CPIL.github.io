import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # Precompute all non-empty subsets of {1..10}
    # Each entry: (sum_mod_101, size, subset_tuple)
    all_subsets = []
    for size in range(1, 11):
        for combo in combinations(range(1, 11), size):
            s = 0
            for x in combo:
                s += inv[x]
            r = s % 101
            all_subsets.append((r, size, combo))
            
    # We want to minimize the maximum size K.
    # Binary search on K from 1 to 10.
    
    def can_match(K):
        # Filter subsets with size <= K
        valid_subsets = [s for s in all_subsets if s[1] <= K]
        
        # Bipartite matching
        # Left: residues 0..100
        # Right: indices of valid_subsets
        # adjacency list for left nodes
        adj = [[] for _ in range(101)]
        for idx, (r, sz, combo) in enumerate(valid_subsets):
            adj[r].append(idx)
            
        match_right = [-1] * len(valid_subsets)
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_right[v] = u
                    return True
            return False
            
        match_count = 0
        for u in range(101):
            visited = [False] * len(valid_subsets)
            if dfs(u, visited):
                match_count += 1
            if match_count == 101:
                return True, match_right, valid_subsets
            if time.time() > deadline:
                return False, None, None
                
        return False, None, None

    best_result = None
    for K in range(1, 11):
        if time.time() > deadline:
            break
        success, match_right, valid_subsets = can_match(K)
        if success:
            # Construct result
            result = [None] * 101
            for v_idx, u in enumerate(match_right):
                if u != -1:
                    result[u] = list(valid_subsets[v_idx][2])
            best_result = result
            break
            
    if best_result is None:
        # Fallback: just use any valid assignment (should not happen)
        # Greedy fallback
        covered = [False] * 101
        result = [None] * 101
        for r, sz, combo in sorted(all_subsets, key=lambda x: x[1]):
            if not covered[r]:
                result[r] = list(combo)
                covered[r] = True
                if all(covered):
                    break
        best_result = result
        
    return best_result
