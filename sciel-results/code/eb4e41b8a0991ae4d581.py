"""Rewritten program for large clique subdivisions in a pinned graph."""
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
    
    # Precompute shortest path distances and next-hop for each pair
    # dist[u][v] = shortest distance, next_hop[u][v] = next vertex on shortest path from u to v
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_hop = [[-1] * n for _ in range(n)]
    
    for src in range(n):
        dist[src][src] = 0
        visited = [False] * n
        queue = deque([src])
        visited[src] = True
        while queue:
            node = queue.popleft()
            for nb in sorted(adj[node]):
                if not visited[nb]:
                    visited[nb] = True
                    dist[nb] = dist[node] + 1
                    next_hop[nb] = node
                    queue.append(nb)
    
    def bfs_path(src, dst, forbidden, max_len=None):
        """Find a path from src to dst avoiding forbidden vertices (except src/dst).
        Returns path or None."""
        if src == dst:
            return [src]
        if src in forbidden or dst in forbidden:
            return None
        # Check if direct edge exists
        if dst in adj[src]:
            if max_len is None or 1 <= max_len:
                return [src, dst]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            if max_len is not None and len(path) > max_len:
                continue
            for nb in sorted(adj[node]):
                if nb == dst:
                    new_path = path + [dst]
                    if max_len is None or len(new_path) - 1 <= max_len:
                        return new_path
                elif nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def find_path_with_constraints(src, dst, forbidden, max_len=None):
        """Find path preferring shorter paths. Returns path or None."""
        if src == dst:
            return [src]
        if src in forbidden or dst in forbidden:
            return None
        # Try direct edge first
        if dst in adj[src]:
            if max_len is None or 1 <= max_len:
                return [src, dst]
        # BFS with length constraint
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            if max_len is not None and len(path) > max_len:
                continue
            for nb in sorted(adj[node]):
                if nb == dst:
                    new_path = path + [dst]
                    if max_len is None or len(new_path) - 1 <= max_len:
                        return new_path
                elif nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def try_build_subdivision(branches, deadline):
        """Try to find internally disjoint paths for all pairs in branches.
        Returns list of paths or None if impossible."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Order pairs: direct edges first, then by shortest path length
        pair_order = []
        for (u, v) in pairs:
            if v in adj[u]:
                pair_order.append((0, 1, u, v))
            else:
                d = dist[u][v]
                pair_order.append((1, d, u, v))
        pair_order.sort()
        
        paths = {}
        used_internal = set()
        
        for _, _, u, v in pair_order:
            if time.time() > deadline:
                return None
            forbidden = branch_set - {u, v} | used_internal
            path = find_path_with_constraints(u, v, forbidden)
            if path is None:
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
    
    def try_add_vertex(current_branches, candidate, deadline):
        """Try to add candidate to current_branches. Returns new branches or None."""
        new_branches = current_branches + [candidate]
        result = try_build_subdivision(new_branches, deadline)
        if result is not None:
            return new_branches, result
        return None
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
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
    
    # Phase 2: Grow from best clique
    if best_size >= 2:
        current_branches = list(best_result['branches'])
    else:
        # Fallback: find an edge
        u, v = P['edges'][0]
        current_branches = [u, v]
        best_size = 2
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Try to grow by adding vertices
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        
        candidates = [v for v in range(n) if v not in set(current_branches)]
        # Sort candidates by degree (higher degree first) and by min distance to existing branches
        def candidate_score(v):
            min_d = min(dist[v][b] for b in current_branches)
            return (-len(adj[v]), min_d)
        
        candidates.sort(key=candidate_score)
        random.shuffle(candidates)  # Add some randomness
        
        added = False
        for v in candidates:
            if time.time() > deadline - 5:
                break
            new_branches, result = try_add_vertex(current_branches, v, deadline)
            if new_branches is not None:
                current_branches = new_branches
                best_size = len(current_branches)
                best_result = {'branches': list(current_branches), 'paths': result}
                added = True
                break
        
        if not added:
            # Try swap: remove one, add two
            if len(current_branches) >= 3:
                swap_done = False
                for i in range(len(current_branches)):
                    if time.time() > deadline - 5:
                        break
                    temp_branches = current_branches[:i] + current_branches[i+1:]
                    candidates2 = [v for v in range(n) if v not in set(temp_branches)]
                    candidates2.sort(key=lambda v: -len(adj[v]))
                    random.shuffle(candidates2)
                    
                    for v in candidates2[:15]:
                        if time.time() > deadline - 5:
                            break
                        new_branches, result = try_add_vertex(temp_branches, v, deadline)
                        if new_branches is not None:
                            # Try to add one more
                            candidates3 = [w for w in range(n) if w not in set(new_branches)]
                            candidates3.sort(key=lambda w: -len(adj[w]))
                            random.shuffle(candidates3)
                            
                            for w in candidates3[:10]:
                                if time.time() > deadline - 5:
                                    break
                                new_branches2, result2 = try_add_vertex(new_branches, w, deadline)
                                if new_branches2 is not None:
                                    current_branches = new_branches2
                                    best_size = len(current_branches)
                                    best_result = {'branches': list(current_branches), 'paths': result2}
                                    swap_done = True
                                    break
                            
                            if not swap_done:
                                current_branches = new_branches
                                best_size = len(current_branches)
                                best_result = {'branches': list(current_branches), 'paths': result}
                            break
                    if swap_done:
                        break
    
    # Phase 3: Random restarts from different cliques
    if time.time() < deadline - 5:
        for clique in cliques[:30]:
            if time.time() > deadline - 5:
                break
            if len(clique) < 2:
                continue
            rb = list(clique)
            random.shuffle(rb)
            
            # Try growing
            for _ in range(40):
                if time.time() > deadline - 5:
                    break
                candidates = [v for v in range(n) if v not in set(rb)]
                candidates.sort(key=lambda v: -len(adj[v]))
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    new_branches, result = try_add_vertex(rb, v, deadline)
                    if new_branches is not None:
                        rb = new_branches
                        if len(rb) > best_size:
                            best_size = len(rb)
                            best_result = {'branches': list(rb), 'paths': result}
                        added = True
                        break
                if not added:
                    break
    
    # Phase 4: Try all pairs of non-adjacent vertices as potential additions
    if time.time() < deadline - 3:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            # Randomly pick two non-branch vertices and try adding them
            non_branches = [v for v in range(n) if v not in set(current_branches)]
            if len(non_branches) < 2:
                break
            v1, v2 = random.sample(non_branches, 2)
            new_branches = current_branches + [v1]
            result = try_build_subdivision(new_branches, deadline)
            if result is not None:
                new_branches = new_branches + [v2]
                result = try_build_subdivision(new_branches, deadline)
                if result is not None:
                    current_branches = new_branches
                    best_size = len(current_branches)
                    best_result = {'branches': list(current_branches), 'paths': result}
    
    if best_result is not None:
        # Ensure types are correct
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    # Final fallback
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
