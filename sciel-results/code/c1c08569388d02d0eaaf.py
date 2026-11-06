import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Inverses of 1..10 mod 101
    invs = [0] * 11
    for i in range(1, 11):
        invs[i] = pow(i, -1, 101)
    
    # Precompute all non-empty subsets and their (size, sum_mod_101, elements)
    all_subsets = []  # list of (size, sum_mod, elements_tuple)
    for k in range(1, 11):
        for combo in combinations(range(1, 11), k):
            s = 0
            for x in combo:
                s += invs[x]
            s %= 101
            all_subsets.append((k, s, combo))
    
    # For each residue, collect all subsets that sum to it, sorted by size
    residue_subsets = {r: [] for r in range(101)}
    for size, s, elements in all_subsets:
        residue_subsets[s].append((size, elements))
    for r in range(101):
        residue_subsets[r].sort(key=lambda x: x[0])
    
    # Try to find minimum max_size k such that we can assign subsets of size <= k to cover all residues
    # Use bipartite matching: residues on left, subsets on right
    # For a fixed k, check if perfect matching exists using only subsets of size <= k
    
    def can_cover(k):
        """Check if all residues can be covered using subsets of size <= k.
        Returns assignment dict if successful, None otherwise."""
        # Build adjacency: residue -> list of subset indices (only those with size <= k)
        # We'll use a simple greedy + augmenting path approach
        
        # For each residue, get candidate subsets (size <= k)
        # Use Hopcroft-Karp style or simple DFS augmenting paths
        
        # Simple approach: try to assign greedily, then fix with augmenting paths
        # Since 101 is small, this should be fast
        
        # Map each subset to an index
        # subset_idx[r] = list of (subset_id, elements) for residue r with size <= k
        subset_for_residue = {}
        for r in range(101):
            subset_for_residue[r] = []
            for size, elements in residue_subsets[r]:
                if size <= k:
                    subset_for_residue[r].append(elements)
        
        # Check if any residue has no candidates
        for r in range(101):
            if len(subset_for_residue[r]) == 0:
                return None
        
        # Use greedy matching with backtracking via augmenting paths
        # match_r[r] = the subset (tuple) assigned to residue r, or None
        # match_s[subset_tuple] = the residue assigned to this subset, or None
        # Note: multiple subsets can have the same tuple? No, each tuple is unique.
        
        # Actually, let's use a proper bipartite matching
        # Left: 101 residues (0-100)
        # Right: all distinct subset tuples with size <= k
        
        # Collect all distinct subsets with size <= k
        all_candidate_subsets = set()
        for r in range(101):
            for elements in subset_for_residue[r]:
                all_candidate_subsets.add(elements)
        
        subset_list = list(all_candidate_subsets)
        subset_to_idx = {s: i for i, s in enumerate(subset_list)}
        num_subsets = len(subset_list)
        
        # Adjacency: residue r -> list of subset indices
        adj = [set() for _ in range(101)]
        for r in range(101):
            for elements in subset_for_residue[r]:
                adj[r].add(subset_to_idx[elements])
        
        # Bipartite matching using DFS augmenting paths
        match_r = [-1] * 101  # match_r[r] = subset index or -1
        match_s = [-1] * num_subsets  # match_s[s] = residue or -1
        
        def dfs(r, visited):
            for s in adj[r]:
                if visited[s]:
                    continue
                visited[s] = True
                if match_s[s] == -1 or dfs(match_s[s], visited):
                    match_s[s] = r
                    match_r[r] = s
                    return True
            return False
        
        # Try to find matching for each residue
        # Use random order or fixed order
        for r in range(101):
            visited = [False] * num_subsets
            if not dfs(r, visited):
                return None
        
        # Build assignment
        assignment = {}
        for r in range(101):
            s_idx = match_r[r]
            if s_idx >= 0:
                assignment[r] = list(subset_list[s_idx])
        return assignment
    
    # Try k from 1 to 10
    best_assignment = None
    for k in range(1, 11):
        if time.time() > deadline:
            break
        result = can_cover(k)
        if result is not None:
            best_assignment = result
            break
    
    # If no assignment found (shouldn't happen), use fallback
    if best_assignment is None:
        # Fallback: use the original greedy approach
        coverage = {}
        for k in range(1, 11):
            for combo in combinations(range(1, 11), k):
                r = 0
                for x in combo:
                    r += invs[x]
                r %= 101
                if r not in coverage:
                    coverage[r] = list(combo)
        best_assignment = {}
        for r in range(101):
            if r in coverage:
                best_assignment[r] = coverage[r]
            else:
                best_assignment[r] = [1]  # fallback
    
    # Construct final result
    result = []
    for r in range(101):
        result.append(best_assignment[r])
    
    return result
