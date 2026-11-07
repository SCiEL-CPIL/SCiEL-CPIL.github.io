"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    
    adj_list = [[] for _ in range(n)]
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj[u][v] = True
        adj[v][u] = True
    
    # Find maximum clique using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P_set, X, max_clique):
        if time.time() > DEADLINE - 10:
            return
        if not P_set and not X:
            if len(R) > len(max_clique[0]):
                max_clique[0] = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj_list[v]) - len(P_set - set(adj_list[v])))
        candidates = [v for v in P_set if not adj[v][pivot]]
        for v in candidates:
            bron_kerbosch(R | {v}, P_set & set(adj_list[v]), X & set(adj_list[v]), max_clique)
            P_set.remove(v)
            X.add(v)
    
    max_clique = [[]]
    bron_kerbosch(set(), set(range(n)), set(), max_clique)
    clique = max_clique[0]
    
    if len(clique) < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    def build_clique_result(clique_verts):
        paths = []
        for pair in itertools.combinations(clique_verts, 2):
            paths.append([int(pair[0]), int(pair[1])])
        return {'branches': [int(v) for v in clique_verts], 'paths': paths}
    
    best_result = build_clique_result(clique)
    
    def try_find_disjoint_paths(branches, time_limit):
        """Backtracking search for internally disjoint paths between all branch pairs."""
        n_b = len(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(range(n_b), 2))
        
        def pair_key(p):
            i, j = p
            bi, bj = branches[i], branches[j]
            if adj[bi][bj]:
                return (0, 0)
            avail = sum(1 for nb in adj_list[bi] if nb not in branch_set) + \
                    sum(1 for nb in adj_list[bj] if nb not in branch_set)
            return (1, -avail)
        
        pairs_sorted = sorted(pairs, key=pair_key)
        pair_idx = {p: i for i, p in enumerate(pairs_sorted)}
        
        paths = [None] * len(pairs_sorted)
        used_interiors = set()
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            if idx == len(pairs_sorted):
                return True
            
            i, j = pairs_sorted[idx]
            bi, bj = branches[i], branches[j]
            
            if adj[bi][bj]:
                paths[idx] = [bi, bj]
                if backtrack(idx + 1):
                    return True
                paths[idx] = None
                return False
            
            forbidden = (branch_set - {bi, bj}) | used_interiors
            
            candidates = []
            queue = deque([(bi, [bi])])
            while queue:
                node, path = queue.popleft()
                if len(path) > 6:
                    continue
                for nb in adj_list[node]:
                    if nb == bj and len(path) < 6:
                        candidates.append(path + [bj])
                    elif nb not in forbidden and nb not in path:
                        queue.append((nb, path + [nb]))
            
            if not candidates:
                return False
            
            candidates.sort(key=len)
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior & used_interiors:
                    continue
                paths[idx] = path
                used_interiors.update(interior)
                if backtrack(idx + 1):
                    return True
                used_interiors.difference_update(interior)
                paths[idx] = None
            
            return False
        
        if backtrack(0):
            result_paths = []
            for pi, pj in pairs:
                idx = pair_idx[(pi, pj)]
                result_paths.append([int(v) for v in paths[idx]])
            return {'branches': [int(v) for v in branches], 'paths': result_paths}
        return None
    
    # Try to extend beyond maximum clique
    current_branches = list(clique)
    
    for attempt in range(50):
        if time.time() > DEADLINE - 5:
            break
        
        best_new = None
        for v in range(n):
            if v in current_branches:
                continue
            if time.time() > DEADLINE - 3:
                break
            
            new_branches = current_branches + [v]
            result = try_find_disjoint_paths(new_branches, time.time() + 2.0)
            if result is not None:
                if len(result['branches']) > len(best_result['branches']):
                    best_result = result
                    current_branches = new_branches
                    best_new = v
                    break
        
        if best_new is None:
            for _ in range(100):
                if time.time() > DEADLINE - 2:
                    break
                sample = random.sample(range(n), len(clique) + 1)
                result = try_find_disjoint_paths(sample, time.time() + 1.0)
                if result is not None and len(result['branches']) > len(best_result['branches']):
                    best_result = result
                    current_branches = sample
                    break
    
    for size in range(len(clique) + 2, min(n, 10) + 1):
        if time.time() > DEADLINE - 3:
            break
        for _ in range(200):
            if time.time() > DEADLINE - 1:
                break
            sample = random.sample(range(n), size)
            result = try_find_disjoint_paths(sample, time.time() + 0.5)
            if result is not None and len(result['branches']) > len(best_result['branches']):
                best_result = result
    
    final_branches = [int(v) for v in best_result['branches']]
    final_paths = [[int(v) for v in path] for path in best_result['paths']]
    
    return {'branches': final_branches, 'paths': final_paths}

# EVOLVE-BLOCK-END
