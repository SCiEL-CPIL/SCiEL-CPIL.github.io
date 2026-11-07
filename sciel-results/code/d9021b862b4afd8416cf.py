# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque
import sys
import random

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute all-pairs shortest paths
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        pred[start][start] = -1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)
    
    def get_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path
    
    def get_all_short_paths(u, v, max_len=6):
        paths = []
        def dfs(curr, end, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == end:
                paths.append(tuple(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, end, path, visited)
                    path.pop()
                    visited.remove(nxt)
        dfs(u, v, [u], {u})
        return paths
    
    def check_and_get_paths(B, deadline):
        B_set = frozenset(B)
        k = len(B)
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))
        
        if not non_adj_pairs:
            return True, {}
        
        domains = {}
        for u, v in non_adj_pairs:
            paths = get_all_short_paths(u, v, max_len=6)
            valid_paths = []
            for p in paths:
                interior = set(p[1:-1])
                if not interior & B_set:
                    valid_paths.append(p)
            if not valid_paths:
                return False, {}
            domains[(u, v)] = valid_paths
        
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        used_interiors = set()
        
        def backtrack(idx, assignment):
            if time.time() > deadline:
                return None
            if idx == len(sorted_pairs):
                return assignment
            
            u, v = sorted_pairs[idx]
            for path in domains[(u, v)]:
                interior = set(path[1:-1])
                if interior & used_interiors:
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                result = backtrack(idx + 1, assignment)
                if result is not None:
                    return result
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return None
        
        result = backtrack(0, {})
        return (result is not None), result
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Find max clique using Bron-Kerbosch
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    best_k = len(best_clique)
    best_B = list(best_clique)
    best_paths = {}
    
    # Verify max clique is valid (it always is)
    ok, paths_dict = check_and_get_paths(best_B, deadline)
    if ok:
        best_paths = paths_dict
    
    vertices_by_degree = sorted(range(n), key=lambda v: -degrees[v])
    random.seed(42)
    
    # Greedy extension
    current_B = list(best_B)
    current_paths = best_paths
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = [v for v in vertices_by_degree if v not in current_B]
        
        for v in candidates:
            if time.time() > deadline - 5:
                break
            new_B = current_B + [v]
            ok, new_paths = check_and_get_paths(new_B, deadline)
            if ok:
                current_B = new_B
                current_paths = new_paths
                if len(current_B) > best_k:
                    best_k = len(current_B)
                    best_B = list(current_B)
                    best_paths = current_paths
                improved = True
                break
    
    # Random restarts to find larger sets
    if time.time() < deadline - 10:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            if best_k + 1 > n:
                break
            B = random.sample(range(n), best_k + 1)
            ok, paths_dict = check_and_get_paths(B, deadline)
            if ok:
                best_k = best_k + 1
                best_B = list(B)
                best_paths = paths_dict
                extended = True
                while extended and time.time() < deadline - 5:
                    extended = False
                    candidates = [v for v in vertices_by_degree if v not in best_B]
                    for v in candidates:
                        if time.time() > deadline - 5:
                            break
                        new_B = list(best_B) + [v]
                        ok, new_paths = check_and_get_paths(new_B, deadline)
                        if ok:
                            best_B = new_B
                            best_paths = new_paths
                            best_k = len(best_B)
                            extended = True
                            break
    
    # Local search: swap to enable growth
    if time.time() < deadline - 8:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            found_swap = False
            for i in range(len(best_B)):
                if time.time() > deadline - 5:
                    break
                for v in vertices_by_degree:
                    if v in best_B:
                        continue
                    new_B = [x for j, x in enumerate(best_B) if j != i] + [v]
                    ok, new_paths = check_and_get_paths(new_B, deadline)
                    if ok:
                        best_B = new_B
                        best_paths = new_paths
                        # Try to extend
                        ext = True
                        while ext and time.time() < deadline - 5:
                            ext = False
                            for w in vertices_by_degree:
                                if w in best_B:
                                    continue
                                nb = list(best_B) + [w]
                                ok2, np2 = check_and_get_paths(nb, deadline)
                                if ok2:
                                    best_B = nb
                                    best_paths = np2
                                    best_k = len(best_B)
                                    ext = True
                                    break
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break
    
    # Construct final result
    B = best_B
    k = len(B)
    paths = []
    
    for i in range(k):
        for j in range(i+1, k):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths.append([u, v])
            else:
                if (u, v) in best_paths:
                    paths.append(list(best_paths[(u, v)]))
                elif (v, u) in best_paths:
                    paths.append(list(reversed(best_paths[(v, u)])))
                else:
                    p = get_path(u, v)
                    if p:
                        paths.append(p)
    
    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
