import time
import random
import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

edge_list = edges

def check_validity(colors, K):
    for u, v in edge_list:
        if colors[u] == colors[v]:
            return False
    used_colors = set(colors)
    if len(used_colors) < 2:
        return True
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            for u, v in edge_list:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        return False
    return True

def get_cost(colors, K):
    cost = 0
    for u, v in edge_list:
        if colors[u] == colors[v]:
            cost += 100
    if cost > 0:
        return cost
    used_colors = set(colors)
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            for u, v in edge_list:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        cost += 10
                        break
    return cost

def get_incremental_cost(colors, K, v, new_c, old_c):
    """Compute cost change when vertex v changes from old_c to new_c."""
    if new_c == old_c:
        return 0
    
    delta = 0
    
    # Properness: check edges incident to v
    for u in adj[v]:
        if colors[u] == new_c:
            delta += 100
        if colors[u] == old_c:
            delta -= 100
    
    if delta > 0:
        return delta
    
    # Bicolored cycle check: only pairs involving old_c or new_c
    # We need to check if changing v's color creates or removes a cycle
    # in bicolored subgraphs involving old_c or new_c
    
    # For each color pair (a, b) where a or b is old_c or new_c,
    # check if the bicolored subgraph has a cycle
    # But we only need to check if the change affects acyclicity
    
    # Simpler: just recompute cost for affected pairs
    affected_pairs = set()
    if old_c < K:
        for c in range(K):
            if c != old_c:
                affected_pairs.add((min(old_c, c), max(old_c, c)))
    if new_c < K:
        for c in range(K):
            if c != new_c:
                affected_pairs.add((min(new_c, c), max(new_c, c)))
    
    # Remove old_c from pair (old_c, new_c) if new_c != old_c
    # Actually the pairs above already handle this
    
    # For each affected pair, check if bicolored subgraph is a forest
    # We need to check the subgraph induced by vertices with colors a and b
    # But only edges where both endpoints have colors a or b
    
    # To be efficient, we check if there's a cycle in the bicolored subgraph
    # for each affected pair
    
    for a, b in affected_pairs:
        uf = UnionFind(N)
        has_cycle = False
        for u, v2 in edge_list:
            cu, cv = colors[u], colors[v2]
            if (cu == a and cv == b) or (cu == b and cv == a):
                if not uf.union(u, v2):
                    has_cycle = True
                    break
        if has_cycle:
            delta += 10
    
    return delta

def local_search(K, time_limit):
    start_time = time.time()
    best_cost = float('inf')
    best_coloring = None
    n_restarts = 0
    max_restarts = 5000
    
    while time.time() - start_time < time_limit and n_restarts < max_restarts:
        n_restarts += 1
        # Random init
        colors = [random.randint(0, K - 1) for _ in range(N)]
        current_cost = get_cost(colors, K)
        
        if current_cost == 0:
            return colors
            
        no_improve = 0
        max_no_improve = 200
        
        while time.time() - start_time < time_limit and no_improve < max_no_improve:
            if current_cost == 0:
                return colors
                
            # Pick vertex: prefer vertices with same-color neighbors or in bicolored cycles
            # Simple heuristic: pick random, but with some bias
            v = random.randint(0, N - 1)
            old_c = colors[v]
            
            best_c = old_c
            min_c = current_cost
            
            for c in range(K):
                if c == old_c:
                    continue
                colors[v] = c
                c_cost = get_cost(colors, K)
                if c_cost < min_c:
                    min_c = c_cost
                    best_c = c
            colors[v] = best_c
            current_cost = min_c
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = list(colors)
                
            if current_cost == 0:
                return colors
                
            if best_c == old_c:
                no_improve += 1
            else:
                no_improve = 0
                
    if best_cost == 0 and best_coloring:
        return best_coloring
    return None

def construct():
    deadline = time.time() + 55.0
    
    # Try K=3
    res = local_search(3, 12.0)
    if res is not None:
        return res
        
    # Try K=4
    remaining = deadline - time.time()
    if remaining > 10:
        res = local_search(4, min(20.0, remaining - 5.0))
        if res is not None:
            return res
            
    # Try K=5
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(5, min(15.0, remaining - 3.0))
        if res is not None:
            return res
            
    # Fallback
    return list(range(N))

# EVOLVE-BLOCK-END
