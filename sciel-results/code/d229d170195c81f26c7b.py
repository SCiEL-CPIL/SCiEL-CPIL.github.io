"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Precompute adjacency
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def fast_is_valid(colors):
    """Fast check: proper coloring + no bicolored cycles."""
    # Proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Group vertices by color
    k = max(colors) + 1
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    # For each pair of colors, check if induced subgraph is a forest
    # Use union-find but only on the relevant vertices
    from itertools import combinations
    for a, b in combinations(range(k), 2):
        if a not in color_groups or b not in color_groups:
            continue
        # Get all vertices with color a or b
        verts = set(color_groups[a]) | set(color_groups[b])
        if len(verts) < 3:
            continue
        
        # Union-find on these vertices
        parent = {}
        for v in verts:
            parent[v] = v
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[ry] = rx
            return True
        
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a and cv == b) or (cu == b and cv == a) or (cu == a and cv == a) or (cu == b and cv == b):
                if not union(u, v):
                    return False
    
    return True

def count_violations_fast(colors):
    """Count violations: proper coloring + bicolored cycles."""
    violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 100
    
    k = max(colors) + 1 if colors else 0
    if k < 2:
        return violations
    
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    from itertools import combinations
    for a, b in combinations(range(k), 2):
        if a not in color_groups or b not in color_groups:
            continue
        verts = set(color_groups[a]) | set(color_groups[b])
        if len(verts) < 3:
            continue
        
        parent = {}
        for v in verts:
            parent[v] = v
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return 1
            parent[ry] = rx
            return 0
        
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a and cv == b) or (cu == b and cv == a) or (cu == a and cv == a) or (cu == b and cv == b):
                violations += union(u, v)
    
    return violations

def get_structural_coloring():
    """Create an initial coloring based on graph structure (toroidal grid)."""
    # The graph looks like a 5x10 toroidal grid
    # Vertex i = row * 10 + col where row in 0..4, col in 0..9
    # Edges: horizontal (col, col+1 mod 10), vertical (row, row+1 mod 5)
    # Let's verify: 0 connects to 1, 7, 43, 49
    # 0 = row 0, col 0
    # 1 = row 0, col 1 (horizontal)
    # 7 = row 0, col 7? No, that doesn't fit
    # Let me re-examine: 0->1 (col+1), 0->7 (?), 0->43 (?), 0->49 (?)
    # Actually looking more carefully at the pattern:
    # Row 0: 0-1-2-3-4-5-6-7-8-9-0 (cycle)
    # But 0 also connects to 7, 43, 49 which breaks the simple grid
    
    # Let me just use a heuristic: color by (i mod k) for various k
    # For k=4: use i % 4
    # For k=5: use i % 5
    # For k=6: use (i // 10) * 2 + (i % 10) % 3  or similar
    
    # Simple approach: use i % k as initial
    return None  # We'll generate random

def local_search_fast(k, initial=None, max_iter=5000, seed=None, deadline=None):
    """Fast local search with incremental violation counting."""
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    current_cost = count_violations_fast(colors)
    best_cost = current_cost
    best_colors = list(colors)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        # Pick a vertex involved in a violation, or random
        v = random.randint(0, n-1)
        old = colors[v]
        best_new = old
        best_c = current_cost
        
        for new_c in range(k):
            if new_c == old:
                continue
            colors[v] = new_c
            nc = count_violations_fast(colors)
            if nc < best_c:
                best_c = nc
                best_new = new_c
        colors[v] = best_new
        current_cost = best_c
    
    if best_cost == 0:
        return best_colors
    return None

def simulated_annealing(k, initial=None, max_iter=10000, seed=None, deadline=None):
    """Simulated annealing for finding valid colorings."""
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    current_cost = count_violations_fast(colors)
    best_cost = current_cost
    best_colors = list(colors)
    
    T = 100.0
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        v = random.randint(0, n-1)
        old = colors[v]
        new_c = random.randint(0, k-1)
        while new_c == old:
            new_c = random.randint(0, k-1)
        
        colors[v] = new_c
        new_cost = count_violations_fast(colors)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_cost = new_cost
        else:
            colors[v] = old
        
        T *= cooling
    
    if best_cost == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s budget
    
    best_result = None
    best_k = 50  # fallback
    
    # Try increasing k from 3
    for k in range(3, 15):
        if time.time() > deadline - 10:
            break
        
        found = None
        
        # Try multiple strategies
        # Strategy 1: Local search with many restarts
        max_restarts = 50
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            seed = random.randint(0, 2**31)
            result = local_search_fast(k, max_iter=2000, seed=seed, deadline=deadline - 3)
            if result is not None:
                found = result
                break
        
        if found is None:
            # Strategy 2: Simulated annealing
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                seed = random.randint(0, 2**31)
                result = simulated_annealing(k, max_iter=5000, seed=seed, deadline=deadline - 3)
                if result is not None:
                    found = result
                    break
        
        if found is not None:
            best_result = found
            best_k = k
            break
    
    if best_result is None:
        # Fallback: use all distinct colors (always valid)
        return list(range(n))
    
    # Ensure Python ints
    return [int(c) for c in best_result]

# EVOLVE-BLOCK-END
