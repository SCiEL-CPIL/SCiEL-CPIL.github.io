"""Rewritten program for large clique subdivisions in a pinned graph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque, defaultdict
from itertools import combinations

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
    
    # Precompute all-pairs shortest paths and predecessors
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)
    
    def get_shortest_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    # Find all simple paths between u and v up to max_len, avoiding a set of forbidden interior vertices
    path_cache = {}
    def get_all_paths(u, v, max_len, forbidden=frozenset()):
        key = (u, v, max_len, forbidden)
        if key in path_cache:
            return path_cache[key]
        
        if u == v:
            path_cache[key] = [[u]]
            return path_cache[key]
            
        paths = []
        def dfs(curr, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                # Check that no interior vertex is in forbidden
                if not any(x in forbidden for x in path[1:-1]):
                    paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited and nxt not in forbidden:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)
        
        dfs(u, [u], {u})
        paths.sort(key=lambda p: (len(p), p))
        path_cache[key] = paths
        return paths

    # Bron-Kerbosch for Maximum Clique
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
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    best_k = 0
    best_B = []
    best_paths = {}
    
    # Check if a set B has a valid clique subdivision
    def check_subdivision(B, time_limit=2.0):
        start_t = time.time()
        B_set = frozenset(B)
        k = len(B)
        
        # Identify non-adjacent pairs
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))
        
        if not non_adj_pairs:
            return True, {}
        
        # Generate candidate paths for each non-adjacent pair
        domains = {}
        for u, v in non_adj_pairs:
            # Try lengths 2, 3, 4, 5, 6
            paths = []
            for l in range(2, 7):
                all_p = get_all_paths(u, v, l, B_set)
                if all_p:
                    paths.extend(all_p)
                    break
            if not paths:
                # Try even longer
                all_p = get_all_paths(u, v, 8, B_set)
                if not all_p:
                    return False, None
                paths = all_p
            
            domains[(u, v)] = paths
        
        # Sort pairs by domain size (MRV heuristic)
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        
        used_interiors = set()
        assignment = {}
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            # Try paths in order of length (shortest first)
            for path in domains[(u, v)]:
                interior = path[1:-1]
                if any(w in used_interiors for w in interior):
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                if backtrack(idx + 1):
                    return True
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return False
        
        if backtrack(0):
            return True, dict(assignment)
        return False, None

    # 1. Start with max clique
    current_B = list(best_clique)
    ok, paths_dict = check_subdivision(current_B, time_limit=5.0)
    if ok:
        best_k = len(current_B)
        best_B = current_B
        best_paths = paths_dict
    
    # 2. Greedy extension: add vertices one at a time
    if best_k < n and time.time() < deadline - 15:
        B_greedy = list(best_B)
        paths_greedy = dict(best_paths)
        
        added = True
        while added and time.time() < deadline - 10:
            added = False
            candidates = [v for v in range(n) if v not in B_greedy]
            # Heuristic: prefer vertices connected to many in B_greedy
            candidates.sort(key=lambda v: -len(adj[v] & set(B_greedy)))
            
            for v in candidates:
                new_B = B_greedy + [v]
                ok, new_paths = check_subdivision(new_B, time_limit=1.5)
                if ok:
                    B_greedy = new_B
                    paths_greedy = new_paths
                    if len(B_greedy) > best_k:
                        best_k = len(B_greedy)
                        best_B = list(B_greedy)
                        best_paths = dict(paths_greedy)
                    added = True
                    break
    
    # 3. Local search with swaps to unlock further extensions
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            B_local = list(best_B)
            paths_local = dict(best_paths)
            
            for i in range(len(B_local)):
                if time.time() > deadline - 5:
                    break
                candidates = [v for v in range(n) if v not in B_local]
                candidates.sort(key=lambda v: -degrees[v])
                
                for v in candidates:
                    new_B = B_local[:i] + [v] + B_local[i+1:]
                    ok, new_paths = check_subdivision(new_B, time_limit=0.8)
                    if ok:
                        B_local = new_B
                        paths_local = dict(new_paths)
                        improved = True
                        break
                if improved:
                    break
            
            # After swaps, try to add a vertex
            if time.time() < deadline - 5:
                candidates = [v for v in range(n) if v not in B_local]
                candidates.sort(key=lambda v: -len(adj[v] & set(B_local)))
                for v in candidates:
                    new_B = B_local + [v]
                    ok, new_paths = check_subdivision(new_B, time_limit=1.5)
                    if ok:
                        B_local = new_B
                        paths_local = new_paths
                        if len(B_local) > best_k:
                            best_k = len(B_local)
                            best_B = list(B_local)
                            best_paths = dict(paths_local)
                        break

    # 4. Random restarts for larger k
    if time.time() < deadline - 5:
        target_k = best_k + 1
        while target_k <= n and time.time() < deadline - 3:
            found = False
            for attempt in range(30):
                if time.time() > deadline - 3:
                    break
                B_rand = random.sample(range(n), target_k)
                ok, paths_rand = check_subdivision(B_rand, time_limit=1.5)
                if ok:
                    best_k = target_k
                    best_B = B_rand
                    best_paths = paths_rand
                    found = True
                    break
            if not found:
                break
            target_k += 1

    # Final verification
    B = best_B
    k = len(B)
    
    if k < 2:
        for u, v in P['edges']:
            if u < v:
                B = [u, v]
                k = 2
                break
        if k < 2:
            B = [0, 1]
            k = 2

    paths = []
    B_set = set(B)
    
    final_ok, final_paths_dict = check_subdivision(B, time_limit=3.0)
    
    if final_ok:
        paths_dict = final_paths_dict
    else:
        # Shrink B until it works
        while len(B) > 2 and not final_ok:
            B.pop()
            final_ok, final_paths_dict = check_subdivision(B, time_limit=2.0)
        if not final_ok:
            for u, v in P['edges']:
                B = [u, v]
                break
            final_ok, final_paths_dict = check_subdivision(B, time_limit=1.0)
            if not final_ok:
                B = [0, 1]
                final_paths_dict = {}
        
        paths_dict = final_paths_dict

    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths.append([u, v])
            else:
                if (u, v) in paths_dict:
                    paths.append(list(paths_dict[(u, v)]))
                elif (v, u) in paths_dict:
                    paths.append(list(reversed(paths_dict[(v, u)])))
                else:
                    p = get_shortest_path(u, v)
                    if p:
                        paths.append(p)
                    else:
                        paths.append([u, v])
    
    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
