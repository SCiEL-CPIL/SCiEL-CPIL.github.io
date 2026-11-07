import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # 1. Find maximum clique using Bron-Kerbosch with pivot
    max_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        if P_set or X:
            pivot = max(list(P_set) + list(X), key=lambda v: len(adj_set[v] & P_set))
            candidates = [v for v in P_set if v not in adj_set[pivot]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch(R + [v], P_set & adj_set[v], X & adj_set[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch([], set(range(n)), set())
    
    # If max clique is large enough, return it directly
    if len(max_clique) >= 7:
        branches = [int(v) for v in max_clique]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}
    
    # 2. Search for clique subdivision of size k using randomized approach
    # We try k from 10 down to 3
    
    # Helper to find disjoint paths for a given set of branches using randomized DFS
    def find_subdivision(branches, max_retries=50):
        branches_set = set(branches)
        pairs = list(combinations(branches, 2))
        num_pairs = len(pairs)
        
        for attempt in range(max_retries):
            if time.time() > deadline:
                return None
            
            # Shuffle pairs to break symmetry
            shuffled_pairs = pairs[:]
            random.shuffle(shuffled_pairs)
            
            # Also shuffle adjacency lists for diversity in path finding
            shuffled_adj = [adj[i][:] for i in range(n)]
            for i in range(n):
                random.shuffle(shuffled_adj[i])
            
            used_vertices = set()
            paths = [None] * num_pairs
            
            def find_paths(idx):
                if time.time() > deadline:
                    return False
                
                if idx == num_pairs:
                    return True
                
                u, v = shuffled_pairs[idx]
                
                # DFS to find a path from u to v avoiding used vertices and other branches
                # except endpoints
                def dfs(curr, path, visited):
                    if time.time() > deadline:
                        return None
                    if curr == v:
                        return path + [v]
                    for nei in shuffled_adj[curr]:
                        if nei in visited:
                            continue
                        if nei in branches_set and nei != v:
                            continue
                        if nei in used_vertices:
                            continue
                        visited.add(nei)
                        result = dfs(nei, path + [nei], visited)
                        if result is not None:
                            return result
                        visited.remove(nei)
                    return None
                
                visited = {u}
                result_path = dfs(u, [u], visited)
                
                if result_path is None:
                    return False
                
                interior = set(result_path[1:-1])
                
                # Check disjointness (should be guaranteed by DFS, but safe)
                if interior & used_vertices:
                    return False
                
                paths[idx] = result_path
                used_vertices.update(interior)
                
                if find_paths(idx + 1):
                    return True
                
                # Backtrack
                used_vertices.difference_update(interior)
                paths[idx] = None
                return False
            
            if find_paths(0):
                # Success
                result_paths = [p for p in paths if p is not None]
                return [list(map(int, p)) for p in result_paths]
        
        return None
    
    # Try different k values from largest to smallest
    for k in range(10, 2, -1):
        if time.time() > deadline:
            break
        
        # Use top vertices by degree as candidates
        # For larger k, use fewer candidates to keep search space manageable
        if k >= 9:
            candidates = vertices_by_degree[:14]
            retries = 30
        elif k >= 7:
            candidates = vertices_by_degree[:18]
            retries = 25
        else:
            candidates = vertices_by_degree[:22]
            retries = 20
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
                
            branches = list(combo)
            paths = find_subdivision(branches, max_retries=retries)
            
            if paths:
                return {
                    'branches': [int(b) for b in branches],
                    'paths': paths
                }
    
    # Fallback: Return max clique found
    if max_clique:
        branches = [int(v) for v in max_clique]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}
    
    # Fallback: Any triangle
    es = {tuple(e) for e in edges}
    for row in combinations(range(n), 3):
        if all(e in es for e in combinations(row, 2)):
            return {
                'branches': [int(u) for u in row],
                'paths': [list(map(int, e)) for e in combinations(row, 2)]
            }
    
    # Absolute fallback
    u, v = edges[0]
    return {
        'branches': [int(u), int(v)],
        'paths': [[int(u), int(v)]]
    }
