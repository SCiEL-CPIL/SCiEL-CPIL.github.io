"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.history.append((rx, ry, self.parent[ry], self.rank[rx], self.rank[ry]))
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def undo(self):
        if not self.history:
            return
        rx, ry, old_parent_ry, old_rank_rx, old_rank_ry = self.history.pop()
        self.parent[ry] = old_parent_ry
        self.rank[rx] = old_rank_rx
        self.rank[ry] = old_rank_ry

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def solve(k, deadline):
        colors = [-1] * n
        uf = [[UF(n) for _ in range(k)] for _ in range(k)]
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == len(order):
                return colors[:]
            
            v = order[idx]
            
            max_color_so_far = -1
            for u in range(idx):
                if colors[order[u]] > max_color_so_far:
                    max_color_so_far = colors[order[u]]
            
            candidates = list(range(min(k, max_color_so_far + 2)))
            
            for c in candidates:
                valid = True
                touched_pairs = []
                
                for u in adj[v]:
                    if colors[u] != -1:
                        d = colors[u]
                        if c == d:
                            valid = False
                            break
                        ci, di = min(c, d), max(c, d)
                        uf_pair = uf[ci][di]
                        if uf_pair.find(v) == uf_pair.find(u):
                            valid = False
                            break
                        touched_pairs.append((ci, di, v, u))
                
                if not valid:
                    continue
                
                for ci, di, vv, uu in touched_pairs:
                    uf[ci][di].union(vv, uu)
                
                colors[v] = c
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                for ci, di, vv, uu in touched_pairs:
                    uf[ci][di].undo()
                
                colors[v] = -1
            
            return None
        
        return backtrack(0)

    start_time = time.time()
    total_time = 80.0
    time_per_k = total_time / 2.0
    
    # Try k=3
    sol = solve(3, start_time + time_per_k)
    if sol:
        return sol
    
    # Try k=4
    sol = solve(4, start_time + total_time)
    if sol:
        return sol
        
    # Fallback
    return list(range(n))
# EVOLVE-BLOCK-END
