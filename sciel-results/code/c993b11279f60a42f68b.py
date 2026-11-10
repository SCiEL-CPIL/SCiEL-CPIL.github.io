import sys
from itertools import combinations
from collections import deque

sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def is_valid_candidate(B):
        B_set = set(B)
        k = len(B)
        if k < 2:
            return False
            
        # Check all non-empty proper subsets S of B
        # Optimization: Only check subsets of size 1, 2, ..., k//2
        # Due to symmetry, checking S and B\S is redundant in terms of req_flow, 
        # but the cut direction matters. 
        # However, Max Flow is symmetric in capacity if we swap source/sink? 
        # The graph is directed in flow, but undirected in original.
        # So MaxFlow(S, T) == MaxFlow(T, S).
        # So we only need to check masks < 2^(k-1)
        
        for mask in range(1, 1 << (k - 1)):
            S = []
            T_part = []
            for i, b in enumerate(B):
                if mask & (1 << i):
                    S.append(b)
                else:
                    T_part.append(b)
            
            req_flow = len(S) * len(T_part)
            
            # Build Flow Network
            num_nodes = 2 * n + 2
            source = 2 * n
            sink = 2 * n + 1
            
            flow_adj = [[] for _ in range(num_nodes)]
            
            def add_edge(u, v, cap):
                flow_adj[u].append([v, cap, len(flow_adj[v])])
                flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
            
            for v in range(n):
                if v in B_set:
                    cap = 1000 
                else:
                    cap = 1
                add_edge(v, v + n, cap)
                
            for u, v in edges:
                add_edge(u + n, v, 1000)
                add_edge(v + n, u, 1000)
                
            for b in S:
                add_edge(source, b, 1000)
                
            for b in T_part:
                add_edge(b + n, sink, 1000)
                
            # Max Flow
            max_flow = 0
            while True:
                parent = [-1] * num_nodes
                parent[source] = -2
                q = deque([source])
                
                while q and parent[sink] == -1:
                    u = q.popleft()
                    for e in flow_adj[u]:
                        v, cap, rev = e
                        if cap > 0 and parent[v] == -1:
                            parent[v] = u
                            if v == sink:
                                break
                            q.append(v)
                
                if parent[sink] == -1:
                    break
                    
                path_flow = 10**9
                v = sink
                while v != source:
                    u = parent[v]
                    for e in flow_adj[u]:
                        if e[0] == v and e[1] > 0:
                            path_flow = min(path_flow, e[1])
                            break
                    v = u
                    
                v = sink
                while v != source:
                    u = parent[v]
                    for i, e in enumerate(flow_adj[u]):
                        if e[0] == v and e[1] > 0:
                            flow_adj[u][i][1] -= path_flow
                            flow_adj[e[2]][1] += path_flow
                            break
                    v = u
                    
                max_flow += path_flow
                if max_flow >= req_flow:
                    break # Early exit if we already met the requirement
                
            if max_flow < req_flow:
                return False
                
        return True

    def construct_paths(B):
        B_set = set(B)
        used_internal = set()
        paths = {}
        pairs = list(combinations(B, 2))
        
        def find_path(u, v, blocked):
            visited = {u}
            q = deque([(u, [u])])
            while q:
                curr, path = q.popleft()
                if curr == v:
                    return path
                for nei in adj[curr]:
                    if nei in visited:
                        continue
                    if nei in B_set and nei != v:
                        continue
                    if nei in blocked:
                        continue
                    visited.add(nei)
                    q.append((nei, path + [nei]))
            return None

        def solve(idx, used_int, current_paths):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            path = find_path(u, v, used_int)
            if path is None:
                return False
            
            internals = set(path[1:-1])
            new_used = used_int | internals
            
            current_paths[(u, v)] = path
            if solve(idx + 1, new_used, current_paths):
                return True
            
            del current_paths[(u, v)]
            return False

        all_paths = {}
        if solve(0, set(), all_paths):
            res_paths = []
            for u, v in pairs:
                res_paths.append(all_paths[(u, v)])
            return res_paths
        else:
            return None

    best_B = None
    # Try k from 7 down to 2
    for k in range(min(7, len(vertices)), 1, -1):
        found = False
        # Use top 12 vertices
        M = 12
        for combo in combinations(vertices[:M], k):
            B = list(combo)
            if is_valid_candidate(B):
                paths = construct_paths(B)
                if paths is not None:
                    best_B = B
                    found = True
                    break
        
        if found:
            break
            
    if best_B is None or len(best_B) < 2:
        # Fallback to Max Clique
        max_clique = []
        def bron_kerbosch(R, P, X):
            nonlocal max_clique
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            if P:
                u = list(P)[0]
            else:
                u = list(X)[0]
            candidates = [v for v in P if v not in adj[u]]
            for v in candidates:
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        bron_kerbosch(set(), set(range(n)), set())
        if not max_clique:
            max_clique = [0]
        if len(max_clique) < 2:
            for u, v in edges:
                max_clique = [u, v]
                break
        best_B = list(max_clique)
        paths = []
        for u, v in combinations(best_B, 2):
            paths.append([u, v])
        return {'branches': best_B, 'paths': paths}
        
    paths = construct_paths(best_B)
    if paths is None:
        # Fallback
        max_clique = []
        def bron_kerbosch(R, P, X):
            nonlocal max_clique
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            if P:
                u = list(P)[0]
            else:
                u = list(X)[0]
            candidates = [v for v in P if v not in adj[u]]
            for v in candidates:
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bron_kerbosch(set(), set(range(n)), set())
        if not max_clique:
            max_clique = [0]
        if len(max_clique) < 2:
            for u, v in edges:
                max_clique = [u, v]
                break
        best_B = list(max_clique)
        paths = []
        for u, v in combinations(best_B, 2):
            paths.append([u, v])

    return {'branches': best_B, 'paths': paths}

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
