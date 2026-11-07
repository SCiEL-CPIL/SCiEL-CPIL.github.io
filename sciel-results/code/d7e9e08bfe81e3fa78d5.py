import time
import random
from itertools import combinations
from collections import deque
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all-pairs shortest paths for distance-based ordering
    D = [[n] * n for _ in range(n)]
    for v in range(n):
        D[v][v] = 0
        q = deque([v])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if D[v][w] > D[v][u] + 1:
                    D[v][w] = D[v][u] + 1
                    q.append(w)

    def find_vertex_disjoint_paths(branches, time_limit=2.0):
        """
        Find internally vertex-disjoint paths between all pairs of branches using max-flow.
        Returns dict mapping (u,v) to path, or None if not possible.
        """
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        pairs.sort(key=lambda p: D[p[0]][p[1]])
        
        start_time = time.time()
        
        # We'll use a greedy approach with flow verification
        # For each pair, find a path that doesn't conflict with previously assigned paths
        used = set()
        paths = {}
        
        def find_path_avoiding(src, dst, forbidden, max_len=10):
            """Find a simple path from src to dst avoiding forbidden vertices."""
            if src == dst:
                return None
            if dst in forbidden:
                return None
            
            dist = {src: 0}
            q = deque([src])
            parent = {src: None}
            
            while q:
                u = q.popleft()
                if u == dst:
                    break
                for w in adj[u]:
                    if w in forbidden:
                        continue
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        q.append(w)
            
            if dst not in dist:
                return None
            
            # Reconstruct path
            path = []
            cur = dst
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        
        def backtrack(idx, used_set, paths_dict):
            if time.time() - start_time > time_limit:
                return False
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forb = branch_set - {u, v} | used_set
            
            # Find candidate paths
            candidates = []
            # Try BFS for shortest path first
            p = find_path_avoiding(u, v, forb)
            if p:
                candidates.append(p)
            
            # Try a few more paths if needed
            if not candidates:
                return False
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior & used_set:
                    continue
                if interior & branch_set:
                    continue
                
                paths_dict[(u, v)] = path
                new_used = used_set | interior
                
                if backtrack(idx + 1, new_used, paths_dict):
                    return True
                
                del paths_dict[(u, v)]
            
            return False
        
        if backtrack(0, set(), {}):
            return {
                'branches': branches,
                'paths': [[int(x) for x in paths[p]] for p in pairs]
            }
        return None

    # Find maximum clique using Bron-Kerbosch with pivoting
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(n)), set())
    
    best_result = None
    best_size = 0
    
    # Check Max Clique
    if len(best_clique) >= 2:
        res = find_vertex_disjoint_paths(sorted(best_clique), time_limit=5.0)
        if res:
            best_result = res
            best_size = len(best_clique)

    # Try to expand beyond max clique size
    degrees = [len(adj[i]) for i in range(n)]
    high_deg = sorted(range(n), key=lambda x: -degrees[x])
    pool = high_deg[:25]
    
    # Try sizes larger than max clique
    max_k_to_try = min(n, best_size + 4)
    
    for k in range(max_k_to_try, best_size, -1):
        if time.time() > deadline - 5:
            break
        if k <= best_size:
            continue
            
        # Strategy 1: Expand Max Clique with high-degree vertices
        if len(best_clique) >= 2:
            extras_needed = k - len(best_clique)
            if 0 < extras_needed <= 4:
                available = [v for v in pool if v not in best_clique]
                if len(available) >= extras_needed:
                    for combo in combinations(available, extras_needed):
                        if time.time() > deadline - 2:
                            break
                        cand = sorted(list(best_clique) + list(combo))
                        res = find_vertex_disjoint_paths(cand, time_limit=1.0)
                        if res:
                            return res

        # Strategy 2: Random sampling from pool
        if time.time() < deadline - 10:
            for _ in range(30):
                if time.time() > deadline - 2:
                    break
                cand = sorted(random.sample(pool, k))
                res = find_vertex_disjoint_paths(cand, time_limit=0.5)
                if res:
                    return res

    if best_result:
        return best_result
        
    # Fallback: return any valid 2-branch solution
    for u, v in P['edges']:
        if u in adj[v]:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}
