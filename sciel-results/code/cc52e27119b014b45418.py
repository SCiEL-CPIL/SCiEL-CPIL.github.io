import time
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bicolor_has_cycle(coloring, adj, c1, c2):
    """Check if subgraph induced by colors c1,c2 has a cycle."""
    n = P['n']
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    for u in range(n):
        cu = coloring[u]
        if cu != c1 and cu != c2:
            continue
        for v in adj[u]:
            if v <= u:
                continue
            cv = coloring[v]
            if cv != c1 and cv != c2:
                continue
            if union(u, v):
                return True
    return False

def _check_all_pairs(coloring, adj, k):
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            if _bicolor_has_cycle(coloring, adj, c1, c2):
                return False
    return True

def _is_valid(coloring, adj, k):
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    return _check_all_pairs(coloring, adj, k)

def _search(k, adj, deadline):
    n = P['n']
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    coloring = [-1] * n
    result = [None]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if result[0] is not None:
            return True
        if idx == n:
            if _is_valid(coloring, adj, k):
                result[0] = list(coloring)
                return True
            return False
        
        v = order[idx]
        forbidden = set()
        for u in adj[v]:
            if coloring[u] >= 0:
                forbidden.add(coloring[u])
        
        for c in range(k):
            if c in forbidden:
                continue
            coloring[v] = c
            
            # Incremental bicolored cycle check
            ok = True
            used = set(coloring[i] for i in range(n) if coloring[i] >= 0)
            for c2 in used:
                if c2 != c and _bicolor_has_cycle(coloring, adj, c, c2):
                    ok = False
                    break
            if not ok:
                coloring[v] = -1
                continue
            
            if backtrack(idx + 1):
                return True
            coloring[v] = -1
        
        return False
    
    backtrack(0)
    return result[0]

def construct():
    deadline = time.time() + 80
    adj = _build_adjacency()
    
    for k in range(2, 50):
        if time.time() > deadline:
            break
        res = _search(k, adj, deadline)
        if res is not None:
            return [int(x) for x in res]
    
    # Fallback: greedy
    n = P['n']
    coloring = [0] * n
    for i in range(n):
        used = set()
        for j in adj[i]:
            if j < i:
                used.add(coloring[j])
        c = 0
        while c in used:
            c += 1
        coloring[i] = c
    return [int(x) for x in coloring]
