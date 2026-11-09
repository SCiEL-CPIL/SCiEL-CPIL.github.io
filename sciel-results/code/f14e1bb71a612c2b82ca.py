import sys
from collections import defaultdict

def has_independent_set_of_size_k(adj, n, k):
    """
    Checks if there is an independent set of size k in the graph.
    Uses a simple backtracking with pruning.
    """
    # Precompute candidates
    # A vertex is a candidate if it can potentially be part of a larger set
    # We search for set of size k
    
    def backtrack(current_set, candidates):
        if len(current_set) == k:
            return True
        
        # Pruning: if current_size + len(candidates) < k, fail
        if len(current_set) + len(candidates) < k:
            return False
        
        for i in range(len(candidates)):
            v = candidates[i]
            # Try adding v
            new_set = current_set + [v]
            
            # Next candidates must be from candidates[i+1:] and not adjacent to v
            new_candidates = []
            for w in candidates[i+1:]:
                if not adj[v][w]:
                    new_candidates.append(w)
            
            if backtrack(new_set, new_candidates):
                return True
            
            # Optional: Branch and bound logic could be added here
            # For now, simple backtracking
            
        return False

    # Initial candidates: all vertices
    # We can order them by degree (descending) to prune faster?
    # Higher degree vertices are less likely to be in independent set?
    # Or sort by index.
    candidates = list(range(n))
    
    return backtrack([], candidates)

def construct():
    # We want to maximize n.
    # Known: n=26 is valid.
    # Try n from 41 down to 27.
    
    for n in range(41, 26, -1):
        # Construct graph
        # Edge (i, j) with i < j if (j - i) % 3 == 1
        adj = [[False] * n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 3 == 1:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append([i, j])
        
        # Check triangle-free
        # Theoretically guaranteed, but let's verify for safety if n is small
        # Actually, proof holds for all n.
        
        # Check independent set of size 10
        if has_independent_set_of_size_k(adj, n, 10):
            # Invalid: has independent set of 10
            continue
        else:
            # Valid: no independent set of 10
            return {'n': n, 'edges': edges}
            
    # Fallback to seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 3 == 1]}
