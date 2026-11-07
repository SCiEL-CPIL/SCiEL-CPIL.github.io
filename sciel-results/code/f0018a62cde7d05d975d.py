"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_clique_subdivision(deadline):
        # Try to find a large clique subdivision
        # Strategy: find a large clique in the graph, then try to extend it
        # by finding internally disjoint paths between non-adjacent pairs
        
        # First, find all cliques up to size n
        # Use a simple backtracking approach to find maximal cliques
        cliques = []
        
        def expand_clique(current, candidates):
            if time.time() > deadline:
                return
            if len(current) >= 2:
                cliques.append(list(current))
            for i, v in enumerate(candidates):
                new_current = current + [v]
                new_candidates = [w for w in candidates[i+1:] if w in adj[v]]
                expand_clique(new_current, new_candidates)
        
        expand_clique([], list(range(n)))
        
        # Sort cliques by size descending
        cliques.sort(key=len, reverse=True)
        
        # For each clique (starting with largest), try to find internally disjoint paths
        for clique in cliques:
            if len(clique) < 2:
                continue
            if time.time() > deadline:
                break
            if len(clique) > 8:  # Limit clique size for tractability
                continue
            
            # For each pair in the clique, we need an internally disjoint path
            # If they're adjacent, use the direct edge
            # Otherwise, find a path in the graph avoiding other branch vertices and previously used internal vertices
            
            # We'll use a greedy approach with backtracking for small cliques
            # For larger cliques, we'll just use direct edges if available
            
            branch_set = set(clique)
            paths = {}
            used_internal = set()
            
            # First, check if all pairs are adjacent (it's a true clique)
            all_adjacent = all((u in adj[v]) for u, v in itertools.combinations(clique, 2))
            
            if all_adjacent:
                # All pairs have direct edges - this is a valid clique subdivision
                for u, v in itertools.combinations(clique, 2):
                    paths[(u, v)] = [u, v]
                return {'branches': list(clique), 'paths': [paths[(u, v)] for u, v in itertools.combinations(clique, 2)]}
            
            # Not all pairs are adjacent - try to find paths
            # This is more complex; for now, only return if it's a true clique
            # Future work: implement path finding for non-adjacent pairs
        
        # If no clique found, return the seed result
        return None
    
    # Set a deadline with safety margin (85 seconds out of 90)
    start_time = time.time()
    deadline = start_time + 85.0
    
    result = find_clique_subdivision(deadline)
    
    if result is not None:
        return result
    
    # Fallback: find any triangle or use seed result
    es = {tuple(e) for e in P['edges']}
    for row in itertools.combinations(range(n), 3):
        if all(e in es for e in itertools.combinations(row, 2)):
            return {'branches': list(row), 'paths': [list(e) for e in itertools.combinations(row, 2)]}
    
    u, v = P['edges'][0]
    return {'branches': [u, v], 'paths': [[u, v]]}
# EVOLVE-BLOCK-END
