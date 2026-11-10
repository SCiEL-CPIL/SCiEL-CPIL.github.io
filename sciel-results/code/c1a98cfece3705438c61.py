"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = _time.time() + 50  # safety margin (90s limit)
    
    def find_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        if src in forbidden and dst in forbidden:
            return None
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None
    
    def bipartite_match_length2(branches, bset):
        """
        For each pair (u,v) that shares at least one common neighbor not in bset,
        try to assign distinct common neighbors via bipartite matching.
        Pairs that are directly adjacent get path [u,v] (no interior).
        Returns dict pair->path or None if infeasible.
        """
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
        
        # Identify pairs needing interior vertices
        need_interior = []  # (u, v, common_neighbors_set)
        direct_pairs = []   # (u, v) where u-v is an edge
        
        for u, v in pairs:
            if v in adj[u]:
                direct_pairs.append((u, v))
            else:
                common = (adj[u] & adj[v]) - bset
                if not common:
                    return None  # No possible path of length 2
                need_interior.append((u, v, common))
        
        if not need_interior:
            # All pairs are direct edges
            return {(u, v): [u, v] for u, v in pairs}
        
        # Build bipartite graph: left = need_interior pairs, right = available interior vertices
        # Left node i corresponds to need_interior[i]
        # Right node j corresponds to vertex j
        # Edge exists if vertex j is a common neighbor of pair i
        
        num_left = len(need_interior)
        
        # For matching, we need: each left node gets exactly one right node,
        # each right node gets at most one left node.
        # This is a standard bipartite matching problem.
        
        # Build adjacency list for left nodes
        left_adj = [set() for _ in range(num_left)]
        for i, (u, v, common) in enumerate(need_interior):
            for w in common:
                left_adj[i].add(w)
        
        # Hopcroft-Karp or simple augmenting path matching
        # Since n is small (24), simple augmenting path is fine
        match_right = {}  # right_vertex -> left_index
        match_left = {}   # left_index -> right_vertex
        
        def try_augment(li, visited):
            for rv in left_adj[li]:
                if rv in visited:
                    continue
                visited.add(rv)
                if rv not in match_right or try_augment(match_right[rv], visited):
                    match_right[rv] = li
                    match_left[li] = rv
                    return True
            return False
        
        for li in range(num_left):
            visited = set()
            if not try_augment(li, visited):
                return None  # Matching infeasible
        
        # Build paths
        paths = {}
        for u, v in direct_pairs:
            paths[(u, v)] = [u, v]
        for i, (u, v, common) in enumerate(need_interior):
            w = match_left[i]
            paths[(u, v)] = [u, w, v]
        
        return paths
    
    def check_branch_set_full(branches):
        """
        Check if a set of branches has valid disjoint paths.
        Strategy:
        1. Try bipartite matching for length-2 paths (fast, works for dense graphs)
        2. If that fails, fall back to sequential BFS with multiple orderings
        Returns paths dict or None.
        """
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
        
        # Strategy 1: Try length-2 matching
        paths = bipartite_match_length2(branches, bset)
        if paths is not None:
            return paths
        
        # Strategy 2: Sequential BFS with multiple orderings
        best_paths = None
        best_total_interior = float('inf')
        
        def pair_key_len(p):
            path = find_shortest_path(p[0], p[1], bset - {p[0], p[1]})
            if path is None:
                return (float('inf'), p)
            return (len(path), p)
        
        orderings = [sorted(pairs, key=pair_key_len)]
        orderings.append(sorted(pairs, key=lambda p: -pair_key_len(p)[0]))
        
        import random
        rng = random.Random(12345)
        for _ in range(3):
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            used_interiors = set()
            paths = {}
            success = True
            for u, v in ordering:
                forbidden = (bset - {u, v}) | used_interiors
                path = find_shortest_path(u, v, forbidden)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & bset:
                    success = False
                    break
                if interior & used_interiors:
                    success = False
                    break
                used_interiors |= interior
                paths[(u, v)] = path
            
            if success:
                total_interior = sum(len(path[1:-1]) for path in paths.values())
                if total_interior < best_total_interior:
                    best_total_interior = total_interior
                    best_paths = paths
        
        return best_paths
    
    def try_grow(initial_branches):
        """Try to grow a branch set greedily."""
        current = list(initial_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            for candidate in range(n):
                if candidate in current:
                    continue
                new_set = current + [candidate]
                paths = check_branch_set_full(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
        return current
    
    # Find a good initial set
    def find_clique():
        deg = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        clique = []
        for v in order:
            if all(v in adj[u] for u in clique):
                clique.append(v)
        return clique
    
    # Try multiple starting points
    best_branches = None
    best_paths = None
    
    # Start with a clique
    clique = find_clique()
    if len(clique) >= 2:
        result = try_grow(clique)
        paths = check_branch_set_full(result)
        if paths is not None:
            best_branches = result
            best_paths = paths
    
    # Also try starting from top-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:8]
    result = try_grow(list(top))
    paths = check_branch_set_full(result)
    if paths is not None and (best_branches is None or len(result) > len(best_branches)):
        best_branches = result
        best_paths = paths
    
    # Try more diverse starts
    import random
    random.seed(42)
    for _ in range(30):
        if _time.time() > deadline:
            break
        start = random.sample(range(n), min(4, n))
        result = try_grow(start)
        paths = check_branch_set_full(result)
        if paths is not None and (best_branches is None or len(result) > len(best_branches)):
            best_branches = result
            best_paths = paths
    
    # Fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
    
    # Format output
    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
