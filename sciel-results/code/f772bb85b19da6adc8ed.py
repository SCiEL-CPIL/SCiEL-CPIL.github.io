import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(colors, edges, n):
    # Check adjacent vertices have different colors
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Check bicolored subgraphs are forests
    # Group vertices by color
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
        
    distinct_colors = list(color_groups.keys())
    
    # For each pair of colors, check if induced subgraph is a forest
    for i in range(len(distinct_colors)):
        for j in range(i+1, len(distinct_colors)):
            c1 = distinct_colors[i]
            c2 = distinct_colors[j]
            
            # Get vertices in this subgraph
            v1 = set(color_groups[c1])
            v2 = set(color_groups[c2])
            sub_vertices = v1.union(v2)
            
            if len(sub_vertices) < 3:
                continue
                
            # Union-Find to detect cycles
            parent = {v: v for v in sub_vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False # Cycle found
                parent[rx] = ry
                return True
            
            has_cycle = False
            for u, v in edges:
                if u in sub_vertices and v in sub_vertices:
                    if not union(u, v):
                        has_cycle = True
                        break
                        
            if has_cycle:
                return False
                
    return True

def _local_search(k, adj, edges, n, time_limit):
    start_time = time.time()
    best_cost = float('inf')
    best_colors = None
    
    # We define cost as number of adjacent conflicts + number of bicolored cycles
    # Actually, we just want to reach 0.
    # Let's use a cost function that is easy to compute incrementally or just recompute.
    # Recomputing is O(E * k^2). For k=3, E=100, this is ~900 ops. Very fast.
    
    def calc_cost(colors):
        cost = 0
        # Adjacent conflicts
        for u, v in edges:
            if colors[u] == colors[v]:
                cost += 10 # Weight conflicts higher
        
        # Bicolored cycles
        color_groups = defaultdict(list)
        for i, c in enumerate(colors):
            color_groups[c].append(i)
            
        distinct_colors = list(color_groups.keys())
        
        for i in range(len(distinct_colors)):
            for j in range(i+1, len(distinct_colors)):
                c1 = distinct_colors[i]
                c2 = distinct_colors[j]
                v1 = set(color_groups[c1])
                v2 = set(color_groups[c2])
                sub_vertices = v1.union(v2)
                if len(sub_vertices) < 3:
                    continue
                parent = {v: v for v in sub_vertices}
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
                cycle_count = 0
                for u, v in edges:
                    if u in sub_vertices and v in sub_vertices:
                        if not union(u, v):
                            cycle_count += 1
                cost += cycle_count * 5 # Weight cycles
                
        return cost

    # Random restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random coloring
        colors = [random.randint(0, k-1) for _ in range(n)]
        current_cost = calc_cost(colors)
        
        if current_cost == 0:
            return colors
        
        temp = 10.0
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random vertex
            v = random.randint(0, n-1)
            old_c = colors[v]
            
            # Pick new color different from old
            new_c = random.randint(0, k-1)
            if new_c == old_c:
                continue
                
            colors[v] = new_c
            new_cost = calc_cost(colors)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                current_cost = new_cost
                if current_cost == 0:
                    return colors
            else:
                colors[v] = old_c
                
            temp *= 0.995
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors[:]
            
    if best_cost == 0:
        return best_colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Try k=3, 4, 5
    for k in [3, 4, 5]:
        res = _local_search(k, adj, edges, n, 20)
        if res is not None:
            # Final verification
            if _check_validity(res, edges, n):
                return [int(c) for c in res]
                
    # Fallback: 6 colors (should always work with i%6)
    res_6 = [i % 6 for i in range(n)]
    if _check_validity(res_6, edges, n):
        return [int(c) for c in res_6]
        
    # Last resort: 7 colors
    return [i % 7 for i in range(n)]
