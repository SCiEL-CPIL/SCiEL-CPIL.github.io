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
    
    def try_build_subdivision(branches, deadline):
        """Try to find internally disjoint paths for all pairs in branches.
        Returns list of paths or None if impossible."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        # Order pairs: direct edges first, then by path length
        pair_order = []
        for (u, v) in pairs:
            if v in adj[u]:
                pair_order.append((0, u, v))
            else:
                pair_order.append((1, u, v))
        pair_order.sort()
        
        paths = {}
        used_internal = set()
        
        for _, u, v in pair_order:
            if time.time() > deadline:
                return None
            forbidden = branch_set - {u, v} | used_internal
            path = bfs_path(u, v, forbidden)
            if path is None:
                return None
            paths[(u, v)] = path
            for x in path[1:-1]:
                used_internal.add(x)
        
        return [paths[(u, v)] for (u, v) in pairs]
    
    def find_max_clique(deadline):
        """Find maximal cliques using Bron-Kerbosch."""
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            # Pivot selection
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(set(), set(range(n)), set())
        cliques.sort(key=len, reverse=True)
        return cliques
    
    start_time = time.time()
    deadline = start_time + 80.0
    
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
    
    if best_result is not None:
        # Try to extend beyond clique size
        pass
    
    # Phase 2: Randomized local search to find larger branch sets
    # Start from the best clique and try adding vertices
    if best_size >= 2:
        base_branches = list(best_result['branches'])
        base_size = best_size
    else:
        # Fallback: find a triangle
        es = {tuple(e) for e in P['edges']}
        base_branches = None
        base_size = 0
        for row in itertools.combinations(range(n), 3):
            if all(e in es or (e[1], e[0]) in es for e in itertools.combinations(row, 2)):
                base_branches = list(row)
                base_size = 3
                break
        if base_branches is None:
            u, v = P['edges'][0]
            base_branches = [u, v]
            base_size = 2
    
    # Try to grow from base_branches
    if time.time() < deadline:
        current_branches = list(base_branches)
        current_size = base_size
        
        # Try adding vertices one at a time
        for _ in range(50):  # limited iterations
            if time.time() > deadline:
                break
            
            # Try each non-branch vertex
            candidates = [v for v in range(n) if v not in set(current_branches)]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                result = try_build_subdivision(new_branches, deadline)
                if result is not None:
                    current_branches = new_branches
                    current_size += 1
                    best_result = {'branches': list(current_branches), 'paths': result}
                    added = True
                    break
            
            if not added:
                # Try removing a vertex and adding two others (swap)
                if len(current_branches) >= 3:
                    for i in range(len(current_branches)):
                        if time.time() > deadline:
                            break
                        temp_branches = current_branches[:i] + current_branches[i+1:]
                        candidates2 = [v for v in range(n) if v not in set(temp_branches)]
                        random.shuffle(candidates2)
                        for v in candidates2[:10]:
                            new_branches = temp_branches + [v]
                            result = try_build_subdivision(new_branches, deadline)
                            if result is not None:
                                current_branches = new_branches
                                best_result = {'branches': list(current_branches), 'paths': result}
                                break
    
    # Phase 3: Random restarts with different starting cliques
    if time.time() < deadline - 5:
        for clique in cliques[:20]:
            if time.time() > deadline - 5:
                break
            if len(clique) < 2:
                continue
            rb = list(clique)
            random.shuffle(rb)
            
            # Try growing
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                candidates = [v for v in range(n) if v not in set(rb)]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    new_branches = rb + [v]
                    result = try_build_subdivision(new_branches, deadline)
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
        # Ensure types are correct
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    # Final fallback
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
