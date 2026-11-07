import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all maximal cliques
    max_cliques = []
    
    def bronk(R, P_set, X):
        if time.time() > deadline:
            return
        if not P_set and not X:
            if R:
                max_cliques.append(list(R))
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    max_cliques.sort(key=lambda c: -len(c))
    
    def find_paths_backtrack(branches):
        """Backtracking search for vertex-disjoint paths between all pairs of branches."""
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)
        
        # Precompute path options for each pair
        path_options = {}
        for u, v in pairs:
            if v in adj[u]:
                path_options[(u, v)] = [[u, v]]
            else:
                forbidden = b_set - {u, v}
                visited = {u}
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue:
                    cur = queue.popleft()
                    if cur == v:
                        path = []
                        node = v
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        path_options[(u, v)] = [path]
                        found = True
                        break
                    for nb in sorted(adj[cur]):
                        if nb not in visited and nb not in forbidden:
                            visited.add(nb)
                            parent[nb] = cur
                            queue.append(nb)
                if not found:
                    return None
                # Find alternative short paths (length 3)
                alternatives = []
                for mid in range(n):
                    if mid in b_set:
                        continue
                    if u in adj[mid] and mid in adj[v]:
                        alternatives.append([u, mid, v])
                if alternatives:
                    path_options[(u, v)] = [path_options[(u, v)][0]] + alternatives
                else:
                    path_options[(u, v)] = [path_options[(u, v)][0]]
        
        best_result = [None]
        
        def bt(idx, used_interior, assigned):
            if time.time() > deadline:
                return
            if idx == num_pairs:
                best_result[0] = dict(assigned)
                return
            u, v = pairs[idx]
            for path in path_options[(u, v)]:
                mids = set(path[1:-1])
                if mids & used_interior:
                    continue
                assigned[(u, v)] = path
                used_interior |= mids
                bt(idx + 1, used_interior, assigned)
                if best_result[0] is not None:
                    return
                used_interior -= mids
        
        bt(0, set(), {})
        return best_result[0]
    
    best_branches = None
    best_paths = None
    best_size = 0
    
    # Try each maximal clique, then extend
    for clique in max_cliques:
        if time.time() > deadline:
            break
        if len(clique) <= best_size:
            continue
        
        paths = find_paths_backtrack(clique)
        if paths is not None and len(clique) > best_size:
            best_size = len(clique)
            best_branches = list(clique)
            best_paths = paths
        
        current = list(clique)
        remaining = [v for v in range(n) if v not in set(current)]
        remaining.sort(key=lambda v: -len(adj[v] & set(current)))
        
        improved = True
        while improved:
            improved = False
            for v in remaining[:]:
                if time.time() > deadline:
                    break
                test = current + [v]
                if len(test) <= best_size:
                    continue
                paths = find_paths_backtrack(test)
                if paths is not None:
                    current = test
                    best_size = len(current)
                    best_branches = list(current)
                    best_paths = paths
                    remaining.remove(v)
                    remaining.sort(key=lambda w: -len(adj[w] & set(current)))
                    improved = True
                    break
    
    # Also try high-degree seed pairs
    if best_size < 5:
        high_deg = sorted(range(n), key=lambda v: -len(adj[v]))
        for i in range(min(10, n)):
            if time.time() > deadline:
                break
            for j in range(i+1, min(10, n)):
                if time.time() > deadline:
                    break
                seed = [high_deg[i], high_deg[j]]
                paths = find_paths_backtrack(seed)
                if paths is None:
                    continue
                current = list(seed)
                remaining = [v for v in range(n) if v not in set(current)]
                remaining.sort(key=lambda v: -len(adj[v] & set(current)))
                improved = True
                while improved:
                    improved = False
                    for v in remaining[:]:
                        if time.time() > deadline:
                            break
                        test = current + [v]
                        if len(test) <= best_size:
                            continue
                        paths = find_paths_backtrack(test)
                        if paths is not None:
                            current = test
                            best_size = len(current)
                            best_branches = list(current)
                            best_paths = paths
                            remaining.remove(v)
                            remaining.sort(key=lambda w: -len(adj[w] & set(current)))
                            improved = True
                            break
    
    if best_branches is None:
        best_branches = [0, 1]
        best_paths = {(0, 1): [0, 1]}
    
    branches = [int(b) for b in sorted(best_branches)]
    paths_list = []
    for u, v in combinations(sorted(best_branches), 2):
        p = best_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
