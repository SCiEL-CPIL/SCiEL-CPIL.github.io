"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Build adjacency list
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute adjacency as a set for quick lookup
adj_set = [set(adj[i]) for i in range(n)]

def has_bicolored_cycle(coloring, k):
    """Check if the subgraph induced by any two colors contains a cycle.
    Uses union-find for each pair of colors."""
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Collect vertices with color c1 or c2
            vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
            if len(vertices) < 3:
                continue
            
            # Union-Find
            parent = {v: v for v in vertices}
            rank = {v: 0 for v in vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # Cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Check edges between vertices of these two colors
            for v in vertices:
                for w in adj[v]:
                    if w > v and coloring[w] in (c1, c2):
                        if not union(v, w):
                            return True  # Found a cycle
    return False

def is_valid(coloring, k):
    """Check proper coloring and no bicolored cycles."""
    # Check proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles
    if has_bicolored_cycle(coloring, k):
        return False
    return True

def count_distinct_colors(coloring):
    return len(set(coloring))

def greedy_coloring(k):
    """Greedy coloring with vertex ordering by degree (descending)."""
    ordering = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    coloring = [-1] * n
    
    for v in ordering:
        # Find available colors
        used = set()
        for u in adj[v]:
            if coloring[u] != -1:
                used.add(coloring[u])
        available = [c for c in range(k) if c not in used]
        if available:
            # Choose color that minimizes future bicolored cycle risk
            # Prefer colors that have fewer existing vertices
            color_counts = defaultdict(int)
            for c in available:
                color_counts[c] = sum(1 for i in range(n) if coloring[i] == c)
            coloring[v] = min(available, key=lambda c: color_counts[c])
        else:
            return None
    
    return coloring

def simulated_annealing(k, initial_coloring, deadline):
    """Simulated annealing to fix bicolored cycles while maintaining proper coloring."""
    coloring = initial_coloring[:]
    if is_valid(coloring, k):
        return coloring
    
    best_coloring = coloring[:]
    best_distinct = count_distinct_colors(coloring)
    
    # Cost function: number of bicolored cycle edges + penalty for non-proper coloring
    def cost(coloring):
        c = 0
        # Non-proper coloring penalty
        for u, v in edges:
            if coloring[u] == coloring[v]:
                c += 1000
        # Bicolored cycle detection via union-find
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
                if len(vertices) < 3:
                    continue
                parent = {v: v for v in vertices}
                rank = {v: 0 for v in vertices}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                for v in vertices:
                    for w in adj[v]:
                        if w > v and coloring[w] in (c1, c2):
                            if not union(v, w):
                                c += 1
                                break
                    else:
                        continue
                    break
        return c
    
    current_cost = cost(coloring)
    T = 100.0
    T_min = 1.0
    cooling = 0.995
    
    while T > T_min and time.time() < deadline:
        # Pick a random vertex to recolor
        v = random.randint(0, n - 1)
        old_color = coloring[v]
        
        # Find valid new colors (different from neighbors)
        used = set(coloring[u] for u in adj[v])
        available = [c for c in range(k) if c not in used and c != old_color]
        
        if not available:
            T *= cooling
            continue
        
        new_color = random.choice(available)
        coloring[v] = new_color
        
        new_cost = cost(coloring)
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < (2.71828 ** (-delta / T)):
            current_cost = new_cost
            if current_cost < cost(best_coloring):
                best_coloring = coloring[:]
        
        coloring[v] = old_color
        T *= cooling
    
    if is_valid(best_coloring, k):
        return best_coloring
    return None

def local_search(k, coloring, deadline):
    """Local search to fix bicolored cycles."""
    coloring = coloring[:]
    if is_valid(coloring, k):
        return coloring
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                if time.time() >= deadline:
                    return None
                # Check if this pair has a cycle
                vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
                if len(vertices) < 3:
                    continue
                parent = {v: v for v in vertices}
                rank = {v: 0 for v in vertices}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                has_cycle = False
                for v in vertices:
                    for w in adj[v]:
                        if w > v and coloring[w] in (c1, c2):
                            if not union(v, w):
                                has_cycle = True
                                break
                    if has_cycle:
                        break
                if not has_cycle:
                    continue
                
                # Try to break the cycle by recoloring a vertex in the cycle
                for v in vertices:
                    if time.time() >= deadline:
                        return None
                    old_color = coloring[v]
                    used = set(coloring[u] for u in adj[v])
                    available = [c for c in range(k) if c not in used and c != old_color]
                    for new_color in available:
                        if new_color in (c1, c2):
                            continue
                        coloring[v] = new_color
                        if not has_bicolored_cycle(coloring, k):
                            if is_valid(coloring, k):
                                return coloring
                            improved = True
                            break
                        coloring[v] = old_color
                    coloring[v] = old_color
                    if improved:
                        break
        if is_valid(coloring, k):
            return coloring
    
    if is_valid(coloring, k):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline with 10s margin
    
    # Try k=2, 3, 4, 5, ... until we find a valid coloring
    for k in range(2, 50):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each k
        best_result = None
        best_distinct = k + 1  # Start with k+1 so any valid result is better
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + k)
            
            # Generate a random initial coloring
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            
            # Try to fix it with greedy repair
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for v in range(n):
                    if time.time() > deadline:
                        break
                    # Check if v is properly colored
                    for u in adj[v]:
                        if coloring[u] == coloring[v]:
                            # Recolor v
                            used = set(coloring[u] for u in adj[v])
                            available = [c for c in range(k) if c not in used]
                            if available:
                                coloring[v] = random.choice(available)
                                improved = True
                            break
            
            # Check if valid
            if is_valid(coloring, k):
                distinct = count_distinct_colors(coloring)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_result = coloring[:]
                break  # Found a valid coloring for this k, no need to try more
            
            # Try simulated annealing
            sa_result = simulated_annealing(k, coloring, deadline)
            if sa_result is not None:
                distinct = count_distinct_colors(sa_result)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_result = sa_result
                break
            
            # Try local search
            ls_result = local_search(k, coloring, deadline)
            if ls_result is not None:
                distinct = count_distinct_colors(ls_result)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_result = ls_result
                break
        
        if best_result is not None:
            # Convert to list of Python ints
            result = [int(c) for c in best_result]
            return result
    
    # Fallback: return all distinct colors (always valid)
    return list(range(n))

# EVOLVE-BLOCK-END
