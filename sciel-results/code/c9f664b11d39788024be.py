import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def is_valid_partial(coloring, v, c):
        for u in adj[v]:
            cu = coloring[u]
            if cu == -1:
                continue
            if cu == c:
                return False
            p1, p2 = (c, cu) if c < cu else (cu, c)
            
            visited = set()
            stack = [u]
            visited.add(u)
            found_path = False
            
            while stack:
                curr = stack.pop()
                if curr == v:
                    found_path = True
                    break
                for w in adj[curr]:
                    if w in visited:
                        continue
                    cw = coloring[w]
                    if cw == -1:
                        continue
                    if (color[curr] == p1 and cw == p2) or (color[curr] == p2 and cw == p1):
                        visited.add(w)
                        stack.append(w)
            if found_path:
                return False
        return True

    def solve_k_colors(k):
        coloring = [-1] * n
        remaining = list(range(n))
        
        def backtrack():
            if time.time() > deadline:
                return False
            if not remaining:
                return True
            
            best_v = -1
            best_available = []
            min_count = k + 1
            
            for v in remaining:
                used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                avail = [c for c in range(k) if c not in used]
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_available = avail
                    if min_count == 0:
                        return False
            
            if not best_available:
                return False
                
            # Shuffle available colors to break symmetry?
            # random.shuffle(best_available) # Might be bad for determinism, but okay
            
            for c in best_available:
                if is_valid_partial(coloring, best_v, c):
                    coloring[best_v] = c
                    remaining.remove(best_v)
                    if backtrack():
                        return True
                    coloring[best_v] = -1
                    remaining.append(best_v)
                    remaining.sort() # Keep order consistent? Or just re-add
            return False
            
        if backtrack():
            return coloring
        return None

    # Try 3 colors
    coloring_3 = solve_k_colors(3)
    if coloring_3:
        if check_full(coloring_3):
            return [int(c) for c in coloring_3]

    # Try 4 colors
    coloring_4 = solve_k_colors(4)
    if coloring_4:
        if check_full(coloring_4):
            return [int(c) for c in coloring_4]

    # Fallback to seed logic
    # ...
    
    return list(range(n))

def check_full(coloring):
    n = P['n']
    edges = P['edges']
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu > cv:
            cu, cv = cv, cu
        pair_edges[(cu, cv)].append((u, v))
        
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True

    for (c1, c2), elist in pair_edges.items():
        uf = UnionFind(n)
        for u, v in elist:
            if not uf.union(u, v):
                return False
    return True
