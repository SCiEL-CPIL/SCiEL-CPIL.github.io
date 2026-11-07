"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85.0  # 90s wall-clock, leave 5s margin
    
    def find_path_with_backtrack(src, dst, forbidden, max_depth=6):
        """Find a simple path from src to dst avoiding forbidden, with backtracking."""
        if src == dst:
            return [src]
        
        def dfs(current, path, visited):
            if len(path) > max_depth + 2:
                return None
            for nb in sorted(adj[current], key=lambda x: (-len(adj[x]), x)):
                if nb == dst and len(path) >= 2:
                    return path + [dst]
                if nb not in forbidden and nb not in visited and nb != src and nb != dst:
                    visited.add(nb)
                    result = dfs(nb, path + [nb], visited)
                    if result is not None:
                        return result
                    visited.remove(nb)
            return None
        
        return dfs(src, [src], {src})
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def try_build_subdivision(branches, deadline, use_backtrack=False):
        """Try to find internally disjoint paths for all pairs in branches.
        Returns list of paths or None if impossible."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Order pairs: direct edges first, then by degree sum (lower degree = more constrained)
        pair_order = []
        for (u, v) in pairs:
            if v in adj[u]:
                pair_order.append((0, 0, u, v))
            else:
                pair_order.append((1, len(adj[u]) + len(adj[v]), u, v))
        pair_order.sort()
        
        paths = {}
        used_internal = set()
        
        for _, _, u, v in pair_order:
            if time.time() > deadline:
                return None
            forbidden = branch_set - {u, v} | used_internal
            
            # Try BFS first (shortest path), then backtrack if needed
            path = bfs_path(u, v, forbidden)
            if path is None and use_backtrack:
                path = find_path_with_backtrack(u, v, forbidden)
            
            if path is None:
                return None
            
            # Verify path validity
            if len(path) < 2:
                return None
            if path[0] != u or path[-1] != v:
                return None
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return None
            for x in path[1:-1]:
                if x in branch_set or x in used_internal:
                    return None
            
            paths[(u, v)] = path
            for x in path[1:-1]:
                used_internal.add(x)
        
        return [paths[(u, v)] for (u, v) in pairs]
    
    def find_max_clique(deadline):
        """Find maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            # Pivot selection: choose vertex in P∪X with max neighbors in P
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        cliques.sort(key=len, reverse=True)
        return cliques
    
    # Phase 1: Find largest clique (guaranteed valid subdivision)
    cliques = find_max_clique(deadline)
    best_result = None
    best_size = 0
    
    for clique in cliques:
        if time.time() > deadline:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        result = try_build_subdivision(clique, deadline)
        if result is not None:
            best_result = {'branches': list(clique), 'paths': result}
            best_size = len(clique)
    
    # Phase 2: Greedy expansion from best clique
    if best_size >= 2:
        base_branches = list(best_result['branches'])
    else:
        # Fallback: find an edge
        u, v = P['edges'][0]
        base_branches = [u, v]
        base_size = 2
        best_size = 2
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Try to grow from base_branches
    current_branches = list(base_branches)
    
    for iteration in range(100):
        if time.time() > deadline - 2:
            break
        
        candidates = [v for v in range(n) if v not in set(current_branches)]
        # Sort candidates by degree (lower degree first - more likely to fit)
        candidates.sort(key=lambda v: len(adj[v]))
        random.shuffle(candidates)
        
        added = False
        for v in candidates:
            if time.time() > deadline - 2:
                break
            new_branches = current_branches + [v]
            result = try_build_subdivision(new_branches, deadline, use_backtrack=True)
            if result is not None:
                current_branches = new_branches
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = {'branches': list(current_branches), 'paths': result}
                added = True
                break
        
        if not added:
            # Try swap: remove one branch, add another
            if len(current_branches) >= 3:
                swapped = False
                for i in range(len(current_branches)):
                    if time.time() > deadline - 2:
                        break
                    temp = current_branches[:i] + current_branches[i+1:]
                    cands2 = [v for v in range(n) if v not in set(temp)]
                    cands2.sort(key=lambda v: len(adj[v]))
                    random.shuffle(cands2)
                    for v in cands2[:15]:
                        if time.time() > deadline - 2:
                            break
                        new_b = temp + [v]
                        result = try_build_subdivision(new_b, deadline, use_backtrack=True)
                        if result is not None:
                            current_branches = new_b
                            if len(current_branches) > best_size:
                                best_size = len(current_branches)
                                best_result = {'branches': list(current_branches), 'paths': result}
                            swapped = True
                            break
                    if swapped:
                        break
    
    # Phase 3: Try all maximal cliques as starting points for expansion
    if time.time() < deadline - 3:
        for clique in cliques[:30]:
            if time.time() > deadline - 3:
                break
            if len(clique) < 2:
                continue
            
            rb = list(clique)
            # Try growing from this clique
            for _ in range(20):
                if time.time() > deadline - 3:
                    break
                candidates = [v for v in range(n) if v not in set(rb)]
                candidates.sort(key=lambda v: len(adj[v]))
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    new_branches = rb + [v]
                    result = try_build_subdivision(new_branches, deadline, use_backtrack=True)
                    if result is not None:
                        rb = new_branches
                        if len(rb) > best_size:
                            best_size = len(rb)
                            best_result = {'branches': list(rb), 'paths': result}
                        added = True
                        break
                if not added:
                    break
    
    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    # Final fallback
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
