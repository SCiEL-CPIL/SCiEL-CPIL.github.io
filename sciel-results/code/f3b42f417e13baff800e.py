import time
import itertools

def construct():
    start_time = time.time()
    LIMIT = 55.0  # safety margin under 60s
    
    p = 101
    denominators = list(range(1, 11))
    n = 10
    
    # Precompute modular inverses
    inverses = [pow(x, -1, p) for x in denominators]
    
    # Generate all non-empty subsets grouped by size
    # For each size, store list of (residue, combo)
    subsets_by_size = {}
    for size in range(1, n + 1):
        subsets_by_size[size] = []
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d - 1]
            residue = s % p
            subsets_by_size[size].append((residue, combo))
    
    # Binary search on maximum allowed size K
    # Check if all 101 residues can be covered by subsets of size <= K
    # using bipartite matching (each residue matched to a distinct subset)
    
    def can_cover(max_size):
        """Check if all residues can be assigned distinct subsets of size <= max_size."""
        # Collect all subsets of size <= max_size
        all_subsets = []
        for sz in range(1, max_size + 1):
            for (res, combo) in subsets_by_size[sz]:
                all_subsets.append((res, combo))
        
        num_subsets = len(all_subsets)
        
        # Build adjacency: for each residue, which subset indices cover it
        adj = [[] for _ in range(p)]
        for idx, (res, combo) in enumerate(all_subsets):
            adj[res].append(idx)
        
        # Hopcroft-Karp style bipartite matching
        # match_subset[s] = residue matched to subset s, or -1
        # match_residue[r] = subset matched to residue r, or -1
        match_subset = [-1] * num_subsets
        match_residue = [-1] * p
        
        # BFS to find shortest augmenting paths
        from collections import deque
        
        def bfs():
            """Find shortest augmenting paths. Returns True if augmenting path exists."""
            dist = [-1] * p  # distance for residues
            queue = deque()
            
            # Initialize: all unmatched residues at distance 0
            for r in range(p):
                if match_residue[r] == -1:
                    dist[r] = 0
                    queue.append(r)
                else:
                    dist[r] = -1
            
            found = False
            while queue:
                r = queue.popleft()
                for s_idx in adj[r]:
                    matched_r = match_subset[s_idx]
                    if matched_r == -1:
                        # Found an augmenting path
                        found = True
                    elif dist[matched_r] == -1:
                        dist[matched_r] = dist[r] + 1
                        queue.append(matched_r)
            
            return found, dist
        
        def dfs(r, dist):
            """Try to find augmenting path from residue r."""
            for s_idx in adj[r]:
                matched_r = match_subset[s_idx]
                if matched_r == -1 or (dist[matched_r] == dist[r] + 1 and dfs(matched_r, dist)):
                    match_subset[s_idx] = r
                    match_residue[r] = s_idx
                    return True
            dist[r] = -1  # Mark as failed
            return False
        
        # Run Hopcroft-Karp
        matching_size = 0
        while True:
            found, dist = bfs()
            if not found:
                break
            for r in range(p):
                if match_residue[r] == -1:
                    if dfs(r, dist):
                        matching_size += 1
        
        if matching_size < p:
            return None
        
        # Build assignment
        assignment = [None] * p
        for r in range(p):
            s_idx = match_residue[r]
            assignment[r] = all_subsets[s_idx][1]
        return assignment
    
    # Binary search for minimum K
    lo, hi = 1, n
    best_assignment = None
    
    for K in range(1, n + 1):
        if time.time() - start_time > LIMIT:
            break
        assignment = can_cover(K)
        if assignment is not None:
            best_assignment = assignment
            break
    
    # Fallback: if no assignment found (shouldn't happen), use greedy
    if best_assignment is None:
        assignments = {}
        for size in range(1, n + 1):
            for (res, combo) in subsets_by_size[size]:
                if res not in assignments:
                    assignments[res] = combo
                if len(assignments) == p:
                    break
            if len(assignments) == p:
                break
            if time.time() - start_time > LIMIT:
                break
        
        result = []
        for r in range(p):
            if r in assignments:
                result.append(list(assignments[r]))
            else:
                result.append(list(denominators))
        return result
    
    result = [list(best_assignment[r]) for r in range(p)]
    
    # Sanity check
    for r in range(p):
        combo = result[r]
        assert len(combo) > 0
        assert len(set(combo)) == len(combo)
        s = sum(inverses[d - 1] for d in combo)
        assert s % p == r, f"Residue mismatch at r={r}: got {s % p}"
        assert all(1 <= d <= 10 for d in combo)
    
    return result
