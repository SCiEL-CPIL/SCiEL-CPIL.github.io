"""Find large clique subdivisions in a pinned graph."""
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_clique(adj, deadline):
    n = len(adj)
    best = []
    
    # Greedy initialization
    def greedy_clique(adj):
        n = len(adj)
        candidates = set(range(n))
        clique = []
        while candidates:
            # Pick vertex with max degree within candidates
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            clique.append(v)
            candidates &= adj[v]
            candidates.discard(v)
        return clique
    
    best = greedy_clique(adj)
    
    # Bron-Kerbosch with pivoting
    def bronk(R, P, X, best_ref):
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > len(best_ref):
                best_ref.extend(R)
            return
        # Pivot
        U = P | X
        # Pick pivot u with max |N(u) ∩ P|
        u = max(U, key=lambda x: len(adj[x] & P)) if U else None
        if u is not None:
            candidates = P - adj[u]
        else:
            candidates = P.copy()
        
        for v in list(candidates):
            bronk(R + [v], P & adj[v], X & adj[v], best_ref)
            P.discard(v)
            X.add(v)
    
    initial_P = set(range(n))
    bronk([], initial_P, set(), best)
    return best

def _find_clique_subdivision(adj, k, deadline):
    """Try to find a K_k subdivision. First try cliques, then search for subdivisions."""
    n = len(adj)
    
    # First: find max clique
    max_clique = _find_max_clique(adj, deadline)
    
    if len(max_clique) >= k:
        branches = max_clique[:k]
        paths = []
        for u, v in combinations(branches, 2):
            paths.append([u, v])
        return branches, paths
    
    # If max clique < k, try to find a K_k subdivision
    # For k=4, we can try: pick 4 vertices, check if there exist vertex-disjoint paths between each pair
    # This is hard in general, but for small graphs we can use a heuristic
    
    if k <= 4 and time.time() < deadline:
        # Try all combinations of k vertices
        # For each, check if we can route vertex-disjoint paths
        # Use a simple approach: for k=4, try to find paths by BFS avoiding other branch paths
        
        def try_subdivision(branches, adj, deadline):
            """Try to find vertex-disjoint paths between all pairs of branches."""
            n = len(adj)
            branches_set = set(branches)
            k = len(branches)
            pairs = list(combinations(range(k), 2))
            
            # We need to assign a path to each pair such that all interior vertices are disjoint
            # and paths are simple and connect the correct endpoints
            
            # Strategy: try different orderings of pairs, and for each pair, 
            # find the shortest path avoiding already-used interior vertices
            # This is a greedy approach that may not find a solution if one exists,
            # but it's fast.
            
            from itertools import permutations
            
            best_paths = None
            for perm in permutations(pairs):
                if time.time() > deadline:
                    break
                used_interiors = set()
                paths = {}
                success = True
                for i, j in perm:
                    bi, bj = branches[i], branches[j]
                    # Find shortest path from bi to bj avoiding used_interiors (except endpoints)
                    # BFS
                    from collections import deque
                    parent = {bi: None}
                    queue = deque([bi])
                    found = False
                    while queue and not found:
                        u = queue.popleft()
                        for v in adj[u]:
                            if v not in parent:
                                if v == bj:
                                    parent[v] = u
                                    found = True
                                    break
                                if v not in branches_set and v not in used_interiors:
                                    parent[v] = u
                                    queue.append(v)
                    if found:
                        # Reconstruct path
                        path = [bj]
                        cur = bj
                        while parent[cur] is not None:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        # Check interior vertices are not in used_interiors
                        interiors = set(path[1:-1])
                        if interiors & used_interiors:
                            success = False
                            break
                        used_interiors |= interiors
                        paths[(i,j)] = path
                    else:
                        success = False
                        break
                if success and all(p in paths for p in pairs):
                    best_paths = {p: paths[p] for p in pairs}
                    break
            
            if best_paths:
                result_paths = []
                for u, v in combinations(branches, 2):
                    # Find which pair index
                    ui = branches.index(u)
                    vi = branches.index(v)
                    key = (min(ui,vi), max(ui,vi))
                    path = best_paths[key]
                    result_paths.append(list(path))
                return list(branches), result_paths
            return None
        
        # Try subsets of size k
        for subset in combinations(range(n), k):
            if time.time() > deadline:
                break
            result = try_subdivision(list(subset), adj, deadline)
            if result:
                return result
    
    # Fallback: return max clique
    branches = max_clique
    paths = []
    for u, v in combinations(branches, 2):
        paths.append([u, v])
    return branches, paths

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Try to find increasingly large clique subdivisions
    # Start with large k and work down
    best_result = None
    
    # First, find max clique size
    max_clique = _find_max_clique(adj, deadline)
    max_clique_size = len(max_clique)
    
    # Try k from min(n, max_clique_size + 2) down to 2
    max_k = min(n, max_clique_size + 3)
    
    for k in range(max_k, 1, -1):
        if time.time() > deadline:
            break
        if k == 2:
            # Always possible: just an edge
            u, v = P['edges'][0]
            branches = [int(u), int(v)]
            paths = [[int(u), int(v)]]
            if best_result is None or len(branches) > len(best_result['branches']):
                best_result = {'branches': branches, 'paths': paths}
            continue
        
        result = _find_clique_subdivision(adj, k, deadline)
        if result:
            branches, paths = result
            # Validate
            if len(branches) == k and len(paths) == k*(k-1)//2:
                # Check all paths are valid
                valid = True
                for path in paths:
                    if len(path) < 2:
                        valid = False
                        break
                    if path[0] not in branches or path[-1] not in branches:
                        valid = False
                        break
                    # Check consecutive vertices are adjacent
                    for i in range(len(path)-1):
                        if path[i+1] not in adj[path[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                # Check interior vertices are disjoint and avoid branches
                if valid:
                    interiors = set()
                    branches_set = set(branches)
                    for path in paths:
                        for v in path[1:-1]:
                            if v in branches_set:
                                valid = False
                                break
                            if v in interiors:
                                valid = False
                                break
                            interiors.add(v)
                        if not valid:
                            break
                if valid:
                    if best_result is None or k > len(best_result['branches']):
                        best_result = {'branches': [int(b) for b in branches], 'paths': [[int(v) for v in p] for p in paths]}
                        # If we found a subdivision larger than max clique, that's great
                        if k > max_clique_size:
                            break  # Can't do better than this for now
    
    if best_result is None:
        # Fallback: return a single edge
        u, v = P['edges'][0]
        best_result = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return best_result
