"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 80  # 80s safety margin
    
    # Precompute: for each pair of vertices, find shortest path (BFS)
    # We'll use this to quickly check feasibility
    
    def bfs_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src, dst)."""
        if src == dst:
            return [src]
        if dst in forbidden and dst != src and dst != dst:
            return None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for w in adj[u]:
                if w not in visited and w not in forbidden:
                    visited.add(w)
                    parent[w] = u
                    queue.append(w)
        return None
    
    def bfs_path_with_forbidden(src, dst, forbidden):
        """Find any path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return [src]
        if src in forbidden or dst in forbidden:
            return None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for w in adj[u]:
                if w not in visited and w not in forbidden:
                    visited.add(w)
                    parent[w] = u
                    queue.append(w)
        return None
    
    def try_clique_subdivision(branches):
        """Try to find vertex-disjoint paths for all pairs in branches.
        Returns dict with 'branches' and 'paths' if successful, else None."""
        branch_set = set(branches)
        k = len(branches)
        if k < 2:
            return None
        
        # For each pair, we need a path. Interior vertices must be disjoint.
        # Strategy: greedily assign paths, trying to use shortest paths first.
        # Order pairs by "difficulty" (fewer available interior vertices first).
        
        # Collect all pairs
        pairs = list(combinations(branches, 2))
        
        # For each pair, compute the set of available interior vertices
        # (all vertices not in branch_set)
        interior_pool = set(range(n)) - branch_set
        
        # Sort pairs by number of common neighbors (fewer = harder)
        def pair_difficulty(p):
            u, v = p
            # Count neighbors of u and v that are not branches
            nu = adj[u] - branch_set
            nv = adj[v] - branch_set
            return len(nu & nv)
        
        pairs.sort(key=pair_difficulty)
        
        # Greedily assign paths
        used_interiors = set()
        paths_dict = {}
        
        for u, v in pairs:
            if time.time() > deadline:
                return None
            
            # Try to find a path from u to v using only unused interior vertices
            forbidden = used_interiors  # can't use already-assigned interior vertices
            # Also can't use any branch vertex as interior
            forbidden_with_branches = forbidden | (branch_set - {u, v})
            
            # Try BFS for shortest path
            path = bfs_path_with_forbidden(u, v, forbidden_with_branches)
            if path is None:
                return None
            
            # Extract interior vertices
            interiors = set(path[1:-1])
            
            # Check no conflicts
            if interiors & used_interiors:
                return None
            if interiors & (branch_set - {u, v}):
                return None
            
            used_interiors |= interiors
            paths_dict[(u, v)] = path
        
        # Build result
        result_paths = []
        for u, v in combinations(branches, 2):
            result_paths.append(list(paths_dict[(u, v)]))
        
        return {'branches': list(branches), 'paths': result_paths}
    
    # Strategy: Try to find large cliques in the graph, then try to extend them
    # by finding vertex-disjoint paths.
    
    # First, find all maximal cliques using Bron-Kerbosch
    def bron_kerbosch(R, P, X, cliques):
        if time.time() > deadline:
            return
        if not P and not X:
            cliques.append(list(R))
            return
        # Pivot selection: choose vertex in P ∪ X with max degree in P
        candidates = list(P | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda v: len(adj[v] & P))
        # Iterate over P \\ N(pivot)
        for v in list(P - adj[pivot]):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], cliques)
            P.remove(v)
            X.add(v)
    
    # Find all cliques
    all_cliques = []
    bron_kerbosch(set(), set(range(n)), set(), all_cliques)
    
    # Sort cliques by size (largest first)
    all_cliques.sort(key=len, reverse=True)
    
    best_result = None
    best_size = 0
    
    # Try each clique as a candidate branch set
    for clique in all_cliques:
        if len(clique) <= best_size:
            break
        if time.time() > deadline:
            break
        
        # Try to find clique subdivision for this clique
        result = try_clique_subdivision(clique)
        if result is not None:
            best_result = result
            best_size = len(clique)
    
    # If no clique worked, try smaller subsets of large cliques
    if best_result is None:
        for clique in all_cliques:
            if len(clique) < 2:
                continue
            if time.time() > deadline:
                break
            # Try subsets of this clique
            for size in range(min(len(clique), best_size + 1), 1, -1):
                if time.time() > deadline:
                    break
                for subset in combinations(clique, size):
                    result = try_clique_subdivision(list(subset))
                    if result is not None:
                        if best_result is None or size > best_size:
                            best_result = result
                            best_size = size
                        break
                if best_result is not None and best_size >= size:
                    break
    
    # Fallback: try all pairs
    if best_result is None:
        for u, v in combinations(range(n), 2):
            if time.time() > deadline:
                break
            result = try_clique_subdivision([u, v])
            if result is not None:
                best_result = result
                best_size = 2
                break
    
    if best_result is None:
        # Absolute fallback
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result
# EVOLVE-BLOCK-END
