"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Find all maximal cliques using Bron-Kerbosch
    def get_maximal_cliques(adj_matrix):
        cliques = []
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if R:
                    cliques.append(frozenset(R))
                return
            # Choose pivot
            if P:
                pivot = next(iter(P))
                candidates = [v for v in P if not adj_matrix[v][pivot]]
            else:
                candidates = list(P)
            for v in candidates:
                bronk(R | {v}, P & {u for u in P if adj_matrix[u][v]}, X & {u for u in X if adj_matrix[u][v]})
                P = P - {v}
                X = X | {v}
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return cliques
    
    # Find all maximal independent sets (cliques in complement)
    def get_maximal_independent_sets(adj_matrix):
        # Build complement
        comp_adj = [[(not adj_matrix[i][j]) and (i != j) for j in range(n)] for i in range(n)]
        return get_maximal_cliques(comp_adj)
    
    max_cliques = get_maximal_cliques(adj)
    max_independent = get_maximal_independent_sets(adj)
    
    # Combine and deduplicate
    all_sets = list(set(max_cliques + max_independent))
    
    # Sort by size descending (larger sets first for better pruning)
    all_sets.sort(key=len, reverse=True)
    
    full_set = frozenset(range(n))
    
    # Iterative deepening DFS for set cover
    def solve(k):
        """Check if we can cover all vertices with k sets."""
        def dfs(uncovered, set_indices, depth):
            if time.time() > deadline:
                return None
            if not uncovered:
                return []
            if depth == 0:
                return None
            if len(set_indices) < depth:
                return None
            
            # Heuristic: pick the vertex in uncovered with fewest available sets that can cover it
            best_vertex = None
            best_count = float('inf')
            for v in uncovered:
                count = 0
                for idx in set_indices:
                    if v in all_sets[idx]:
                        count += 1
                if count < best_count:
                    best_count = count
                    best_vertex = v
            if best_count == 0:
                return None
            
            # Try each set that contains best_vertex
            for i in range(len(set_indices)):
                idx = set_indices[i]
                s = all_sets[idx]
                if best_vertex in s:
                    new_uncovered = uncovered - s
                    if len(new_uncovered) < len(uncovered):
                        new_indices = set_indices[:i] + set_indices[i+1:]
                        result = dfs(new_uncovered, new_indices, depth-1)
                        if result is not None:
                            return [idx] + result
            return None
        
        return dfs(full_set, list(range(len(all_sets))), k)
    
    # Try k = 1, 2, 3, ...
    for k in range(1, n+1):
        if time.time() > deadline:
            break
        result = solve(k)
        if result is not None:
            # Convert indices to actual sets
            return [list(all_sets[idx]) for idx in result]
    
    # Fallback: singletons (always valid)
    return [[i] for i in range(n)]
# EVOLVE-BLOCK-END
