"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _verify(coloring, edges):
    """Verify: adjacent vertices differ, and no bicolored cycle exists."""
    n = len(coloring)
    # Check adjacency
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycle using union-find per color pair
    color_pairs = {}
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            continue
        key = (min(cu, cv), max(cu, cv))
        if key not in color_pairs:
            color_pairs[key] = []
        color_pairs[key].append((u, v))
    
    # Union-find for each color pair
    def has_cycle(n, edge_list):
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edge_list:
            ru, rv = find(u), find(v)
            if ru == rv:
                return True
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
        return False
    
    for key, elist in color_pairs.items():
        if has_cycle(n, elist):
            return False
    return True

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 55
    
    # Direct construction: try specific 3-coloring patterns based on grid structure
    # The graph is a 7x7 toroidal grid with vertex i at (i//7, i%7)
    # Try: color = (row + 2*col) % 3
    patterns = [
        lambda r, c: (r + 2*c) % 3,
        lambda r, c: (2*r + c) % 3,
        lambda r, c: (r + c) % 3,
        lambda r, c: (r - c) % 3,
        lambda r, c: (2*r - c) % 3,
        lambda r, c: (r + 2*c + 1) % 3,
        lambda r, c: (2*r + c + 1) % 3,
        lambda r, c: (r + c + 1) % 3,
        lambda r, c: (r - c + 1) % 3,
        lambda r, c: (r + 2*c + 2) % 3,
        lambda r, c: (2*r + c + 2) % 3,
        lambda r, c: (r + c + 2) % 3,
    ]
    
    for pat in patterns:
        coloring = [pat(i // 7, i % 7) for i in range(n)]
        if _verify(coloring, edges):
            return coloring
    
    # Try 4-coloring patterns
    patterns4 = [
        lambda r, c: (r + 2*c) % 4,
        lambda r, c: (2*r + c) % 4,
        lambda r, c: (r + c) % 4,
        lambda r, c: (r - c) % 4,
        lambda r, c: (2*r - c) % 4,
        lambda r, c: (r + 2*c + 1) % 4,
        lambda r, c: (2*r + c + 1) % 4,
        lambda r, c: (r + c + 1) % 4,
    ]
    for pat in patterns4:
        coloring = [pat(i // 7, i % 7) for i in range(n)]
        if _verify(coloring, edges):
            return coloring
    
    # Fallback: use backtracking for small k
    for k in range(3, 8):
        if time.time() > deadline - 3:
            break
        result = _backtrack_solve(n, adj, edges, k, deadline)
        if result is not None:
            return result
    
    # Last resort
    return list(range(n))

def _backtrack_solve(n, adj, edges, k, deadline):
    """Backtracking solver with union-find rollback."""
    color = [-1] * n
    
    class UF:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
            self.log = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.log.append((rx, ry, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self):
            if not self.log:
                return
            rx, ry, old_rank_rx = self.log.pop()
            self.parent[ry] = ry
            self.rank[rx] = old_rank_rx
    
    ufs = {}
    for i in range(k):
        for j in range(i+1, k):
            ufs[(i, j)] = UF(n)
    
    # Order vertices by degree descending, then by index
    deg = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: (-deg[v], v))
    
    def backtrack(idx):
        if time.time() > deadline - 1:
            return None
        if idx == n:
            return list(color)
        
        v = order[idx]
        for c in range(k):
            ok = True
            for u in adj[v]:
                if color[u] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            color[v] = c
            ufs_rb = []
            valid = True
            for u in adj[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                key = (min(c, cu), max(c, cu))
                if ufs[key].union(v, u):
                    ufs_rb.append(key)
                else:
                    valid = False
                    break
            
            if not valid:
                for key in reversed(ufs_rb):
                    ufs[key].rollback()
                color[v] = -1
                continue
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            for key in reversed(ufs_rb):
                ufs[key].rollback()
            color[v] = -1
        
        return None
    
    return backtrack(0)

# EVOLVE-BLOCK-END
