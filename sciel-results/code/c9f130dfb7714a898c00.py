import time
import sys
from collections import defaultdict

sys.setrecursionlimit(5000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try pattern-based colorings first
    # The graph is roughly a 7x7 grid (vertices 0-48) plus vertex 49
    # Try various linear patterns: color(v) = (a * (v//7) + b * (v%7) + c) mod k
    
    best_coloring = None
    best_k = 50
    
    for k in range(3, 13):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try different pattern parameters
        for a in range(k):
            for b in range(k):
                for c in range(k):
                    coloring = [0] * n
                    for v in range(49):
                        row = v // 7
                        col = v % 7
                        coloring[v] = (a * row + b * col + c) % k
                    # Vertex 49: try all colors
                    for c49 in range(k):
                        coloring[49] = c49
                        if _check_coloring(coloring, edges):
                            if k < best_k:
                                best_k = k
                                best_coloring = list(coloring)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            return [int(c) for c in best_coloring]
    
    if best_coloring is not None:
        return [int(c) for c in best_coloring]
    
    # Fallback: backtracking search
    for k in range(3, 13):
        if time.time() > deadline - 10:
            break
        
        coloring = [-1] * n
        
        class UF:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.log = []
            
            def find(self, x):
                while self.parent[x] != x:
                    x = self.parent[x]
                return x
            
            def union(self, u, v):
                ru = self.find(u)
                rv = self.find(v)
                if ru == rv:
                    return False
                if self.rank[ru] < self.rank[rv]:
                    ru, rv = rv, ru
                self.log.append((rv, ru, self.rank[ru]))
                self.parent[rv] = ru
                if self.rank[ru] == self.rank[rv]:
                    self.rank[ru] += 1
                return True
            
            def rollback(self, snapshot):
                while len(self.log) > snapshot:
                    rv, ru, old_rank = self.log.pop()
                    self.parent[rv] = rv
                    self.rank[ru] = old_rank
        
        pair_to_idx = {}
        idx = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                pair_to_idx[(c1, c2)] = idx
                idx += 1
        num_pairs = idx
        ufs = [UF(n) for _ in range(num_pairs)]
        
        def backtrack(v_idx):
            if time.time() > deadline - 3:
                return False
            if v_idx == n:
                return True
            
            v = v_idx
            for c in range(k):
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[v] = c
                snapshots = []
                ok = True
                
                for nb in adj[v]:
                    if coloring[nb] == -1:
                        continue
                    d = coloring[nb]
                    if d == c:
                        continue
                    c_min, c_max = (c, d) if c < d else (d, c)
                    pi = pair_to_idx[(c_min, c_max)]
                    uf = ufs[pi]
                    snap = len(uf.log)
                    if not uf.union(v, nb):
                        ok = False
                        break
                    snapshots.append((uf, snap))
                
                if ok:
                    if backtrack(v_idx + 1):
                        return True
                
                coloring[v] = -1
                for uf, snap in snapshots:
                    uf.rollback(snap)
            
            return False
        
        if backtrack(0):
            if _check_coloring(coloring, edges):
                return [int(c) for c in coloring]
    
    # Final fallback
    coloring = [0] * n
    for v in range(n):
        used = set(coloring[nb] for nb in adj[v])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    return [int(c) for c in coloring]
