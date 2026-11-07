"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0
    
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        """Find shortest path from u to v avoiding banned vertices (interior)."""
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def format_result(branches, paths_dict):
        """Validate and format the result."""
        if not branches or len(branches) < 2:
            return None
        bset = set(branches)
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if set(paths_dict.keys()) != expected_pairs:
            return None
        
        used_interior = set()
        for fkey, path in paths_dict.items():
            if len(path) < 2:
                return None
            a, b = path[0], path[-1]
            if a not in bset or b not in bset or a == b:
                return None
            interior = set(path[1:-1])
            if interior & bset or interior & used_interior:
                return None
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return None
            used_interior |= interior
            
        out_paths = []
        for a, b in combinations(branches, 2):
            out_paths.append(paths_dict[frozenset((a, b))])
        return {'branches': list(branches), 'paths': out_paths}

    def incremental_search(start_vertices, max_depth, rng):
        """
        Build branch set incrementally. Start with start_vertices, then try to add
        one vertex at a time, checking if disjoint paths exist for all new pairs.
        Returns best (branches, paths_dict) found.
        """
        best = None
        current_branches = list(start_vertices)
        current_paths = {}
        
        # Find paths for initial set
        bset = set(current_branches)
        used_interior = set()
        pairs = list(combinations(current_branches, 2))
        for a, b in pairs:
            banned = used_interior | (bset - {a, b})
            path = find_shortest_path(a, b, banned)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & (bset - {a, b}) or interior & used_interior:
                return None
            used_interior |= interior
            current_paths[frozenset((a, b))] = path
        
        if len(current_branches) >= 2:
            best = (list(current_branches), dict(current_paths))
        
        # Try to add vertices one by one
        depth = 0
        while depth < max_depth and not time_up():
            depth += 1
            bset = set(current_branches)
            # Candidate vertices: not already branches, not in used_interior
            candidates = [v for v in range(n) if v not in bset and v not in used_interior]
            # Sort by degree descending
            candidates.sort(key=lambda x: degrees[x], reverse=True)
            
            added = False
            for new_v in candidates:
                if time_up():
                    break
                
                new_branches = current_branches + [new_v]
                new_bset = set(new_branches)
                
                # Try to find paths for new pairs (new_v, each existing branch)
                new_paths = dict(current_paths)
                new_used = set(used_interior)
                success = True
                
                # Order: pairs involving new_v, sorted by distance
                new_pairs = [(new_v, b) for b in current_branches]
                new_pairs.sort(key=lambda p: len(adj[p[0]] & adj[p[1]]))
                
                for a, b in new_pairs:
                    banned = new_used | (new_bset - {a, b})
                    path = find_shortest_path(a, b, banned)
                    if path is None:
                        success = False
                        break
                    interior = set(path[1:-1])
                    if interior & (new_bset - {a, b}) or interior & new_used:
                        success = False
                        break
                    new_used |= interior
                    new_paths[frozenset((a, b))] = path
                
                if success:
                    current_branches = new_branches
                    current_paths = new_paths
                    used_interior = new_used
                    if best is None or len(current_branches) > len(best[0]):
                        best = (list(current_branches), dict(current_paths))
                    added = True
                    break
            
            if not added:
                break
        
        return best

    def try_full_set(branches, rng):
        """Try to find disjoint paths for a complete set of branches with multiple orderings."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Multiple orderings
        orders = []
        orders.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        orders.append(sorted(pairs, key=lambda p: len(adj[p[0]] & adj[p[1]])))
        for _ in range(3):
            o = list(pairs)
            rng.shuffle(o)
            orders.append(o)
        
        for order in orders:
            used_interior = set()
            paths_dict = {}
            success = True
            for a, b in order:
                banned = used_interior | (bset - {a, b})
                path = find_shortest_path(a, b, banned)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & (bset - {a, b}) or interior & used_interior:
                    success = False
                    break
                used_interior |= interior
                paths_dict[frozenset((a, b))] = path
            if success:
                return paths_dict
        return None

    rng = random.Random(42)
    
    # Strategy 1: Incremental build from high-degree seeds
    best_result = None
    best_size = 0
    
    # Try incremental from top-degree vertices
    for seed_size in range(2, 5):
        if time_up():
            break
        seed = sorted_verts[:seed_size]
        result = incremental_search(seed, 20, rng)
        if result and len(result[0]) > best_size:
            best_size = len(result[0])
            best_result = format_result(result[0], result[1])
    
    # Strategy 2: Try full sets of various sizes
    for k in range(min(14, n), 1, -1):
        if time_up():
            break
        if k <= best_size:
            break
        
        # Generate candidates
        cands = [sorted_verts[:k]]
        for _ in range(5):
            cands.append(rng.sample(range(n), k))
        
        for cand in cands:
            if time_up():
                break
            paths_dict = try_full_set(cand, rng)
            if paths_dict:
                result = format_result(cand, paths_dict)
                if result:
                    best_result = result
                    best_size = k
                    break
        if best_size >= k:
            break
    
    # Fallback
    if best_result is None:
        # Greedy clique
        remaining = set(range(n))
        clique = []
        while remaining:
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            clique.append(v)
            remaining = remaining & adj[v]
        if len(clique) >= 2:
            paths = [[u, v] for u, v in combinations(clique, 2) if v in adj[u]]
            best_result = {'branches': clique, 'paths': paths}
        else:
            u, v = edges[0]
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    return best_result
# EVOLVE-BLOCK-END
