import time
import sys
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_valid(coloring, max_colors):
        """Full validation: proper coloring + no bicolored cycles."""
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored cycles using DSU
        color_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = DSU(n)
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False
        
        used = set(coloring)
        return len(used) <= max_colors
    
    def backtrack(k, max_colors, colors, dsus):
        """Incremental backtracking with DSU for bicolored cycle detection."""
        if time.time() > deadline:
            return None
        
        if k == n:
            return colors[:]
            
        for c in range(max_colors):
            # Check proper coloring
            valid = True
            for u in adj[k]:
                if colors[u] == -1:
                    continue
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check bicolored cycle constraint
            new_dsus = {key: val for key, val in dsus.items()}
            cycle_found = False
            
            for u in adj[k]:
                if colors[u] == -1:
                    continue
                d = colors[u]
                if d == c:
                    continue
                
                pair = (min(c, d), max(c, d))
                if pair not in new_dsus:
                    new_dsus[pair] = DSU(n)
                
                if not new_dsus[pair].union(k, u):
                    cycle_found = True
                    break
                    
            if cycle_found:
                continue
                
            colors[k] = c
            res = backtrack(k + 1, max_colors, colors, new_dsus)
            if res is not None:
                return res
            colors[k] = -1
            
        return None
    
    def local_search(coloring, max_colors, max_iters=2000):
        """Try to reduce color count by re-coloring vertices."""
        if time.time() > deadline:
            return coloring, len(set(coloring))
        
        current = coloring[:]
        current_count = len(set(current))
        best = current[:]
        best_count = current_count
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if current_count <= 3:
                break
            
            # Pick a random vertex to re-color
            v = random.randint(0, n - 1)
            old_c = current[v]
            
            # Try to re-color with a color that reduces total distinct colors
            used_colors = set(current)
            target_colors = [c for c in used_colors if c != old_c]
            
            if not target_colors:
                continue
            
            random.shuffle(target_colors)
            
            for new_c in target_colors:
                # Check if new_c is valid for v
                valid = True
                for u in adj[v]:
                    if current[u] == new_c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Tentatively re-color
                current[v] = new_c
                
                # Check bicolored cycles
                valid_bc = True
                for c1 in used_colors:
                    if c1 == new_c:
                        continue
                    for c2 in used_colors:
                        if c2 <= c1 or c2 == new_c:
                            continue
                        pair = (min(c1, c2), max(c1, c2))
                        uf = DSU(n)
                        for u, w in edges:
                            cu, cw = current[u], current[w]
                            if (cu == c1 and cw == c2) or (cu == c2 and cw == c1):
                                if not uf.union(u, w):
                                    valid_bc = False
                                    break
                        if not valid_bc:
                            break
                    if not valid_bc:
                        break
                
                if valid_bc:
                    new_count = len(set(current))
                    if new_count < best_count:
                        best = current[:]
                        best_count = new_count
                    # Keep the change if it doesn't increase count
                    current_count = new_count
                    break
                else:
                    current[v] = old_c
        
        return best, best_count
    
    # Strategy 1: Sequential backtracking with increasing color limits
    for mc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        colors = [-1] * n
        dsus = {}
        res = backtrack(0, mc, colors, dsus)
        if res is not None:
            # Try to improve with local search
            improved, count = local_search(res, mc)
            if count < len(set(res)):
                return [int(x) for x in improved]
            return [int(x) for x in res]
    
    # Strategy 2: Random restarts with greedy coloring + local search
    best_coloring = list(range(n))
    best_count = n
    
    for max_colors in [4, 5, 6]:
        if time.time() > deadline:
            break
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7 + 13)
            order = list(range(n))
            random.shuffle(order)
            
            coloring = [-1] * n
            valid = True
            
            for v in order:
                used_colors = set(coloring[u] for u in adj[v] if coloring[u] >= 0)
                available = [c for c in range(max_colors) if c not in used_colors]
                
                if not available:
                    valid = False
                    break
                
                coloring[v] = random.choice(available)
            
            if not valid:
                continue
            
            if check_valid(coloring, max_colors):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                
                # Try local search to improve
                improved, count = local_search(coloring, max_colors)
                if count < best_count:
                    best_count = count
                    best_coloring = improved[:]
                
                if best_count <= 3:
                    return [int(x) for x in best_coloring]
    
    # Strategy 3: More aggressive local search from random initial colorings
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        random.seed(1000 + attempt)
        coloring = [random.randint(0, 4) for _ in range(n)]
        
        # First, fix proper coloring violations
        for _ in range(50):
            changed = False
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    # Re-color v
                    used = set(coloring[x] for x in adj[v])
                    available = [c for c in range(5) if c not in used]
                    if available:
                        coloring[v] = random.choice(available)
                        changed = True
            if not changed:
                break
        
        if check_valid(coloring, 5):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
            
            improved, count = local_search(coloring, 5)
            if count < best_count:
                best_count = count
                best_coloring = improved[:]
    
    # Final verification
    if not check_valid(best_coloring, n):
        # Fallback: use all distinct colors
        return list(range(n))
    
    result = [int(c) for c in best_coloring]
    
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    return result
