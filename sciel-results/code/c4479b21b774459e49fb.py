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
adj_set = [set() for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)
    adj_set[u].add(v)
    adj_set[v].add(u)

# Edge set for fast lookup
edge_set = set()
for u, v in edges:
    if u < v:
        edge_set.add((u, v))
    else:
        edge_set.add((v, u))

def has_bicolored_cycle(colors):
    """Check if any pair of colors forms a cycle in the induced subgraph."""
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = list(color_groups.keys())
    
    for ai in range(len(used_colors)):
        for bi in range(ai + 1, len(used_colors)):
            a, b = used_colors[ai], used_colors[bi]
            
            # Collect vertices with color a or b
            relevant_vertices = set(color_groups[a]) | set(color_groups[b])
            
            # Collect edges where both endpoints have color a or b
            relevant_edges = []
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a or cu == b) and (cv == a or cv == b):
                    relevant_edges.append((u, v))
            
            if len(relevant_edges) < 3:
                continue
            
            # Union-Find to detect cycle
            parent = {v: v for v in relevant_vertices}
            rank = {v: 0 for v in relevant_vertices}
            
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
            
            for u, v in relevant_edges:
                if not union(u, v):
                    return True
    
    return False

def find_bicolored_cycles(colors):
    """Find all bicolored cycles and return list of (color_a, color_b, cycle_vertices)."""
    cycles = []
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = list(color_groups.keys())
    
    for ai in range(len(used_colors)):
        for bi in range(ai + 1, len(used_colors)):
            a, b = used_colors[ai], used_colors[bi]
            
            relevant_vertices = set(color_groups[a]) | set(color_groups[b])
            if len(relevant_vertices) < 3:
                continue
            
            relevant_edges = []
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a or cu == b) and (cv == a or cv == b):
                    relevant_edges.append((u, v))
            
            if len(relevant_edges) < 3:
                continue
            
            parent = {v: v for v in relevant_vertices}
            rank = {v: 0 for v in relevant_vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return rx
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return rx
            
            cycle_found = False
            cycle_vertices = []
            for u, v in relevant_edges:
                root = union(u, v)
                if root is not None and find(u) == find(v):
                    cycle_found = True
                    # Collect vertices in this component
                    comp = set()
                    for vv in relevant_vertices:
                        if find(vv) == root:
                            comp.add(vv)
                    cycle_vertices = list(comp)
                    break
            
            if cycle_found and cycle_vertices:
                cycles.append((a, b, cycle_vertices))
    
    return cycles

def is_valid(colors):
    """Check proper coloring and no bicolored cycles."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return not has_bicolored_cycle(colors)

def count_violations_fast(colors):
    """Fast count of violations."""
    violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            violations += 100
    
    if violations > 0:
        return violations
    
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    
    used_colors = list(color_groups.keys())
    
    for ai in range(len(used_colors)):
        for bi in range(ai + 1, len(used_colors)):
            a, b = used_colors[ai], used_colors[bi]
            
            relevant_edges = []
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a or cu == b) and (cv == a or cv == b):
                    relevant_edges.append((u, v))
            
            if len(relevant_edges) < 3:
                continue
            
            relevant_vertices = set()
            for u, v in relevant_edges:
                relevant_vertices.add(u)
                relevant_vertices.add(v)
            
            parent = {v: v for v in relevant_vertices}
            rank = {v: 0 for v in relevant_vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return 1
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return 0
            
            for u, v in relevant_edges:
                violations += union(u, v)
    
    return violations

def break_cycle_recolor(colors, k, cycle_vertices, color_a, color_b, deadline=None):
    """Try to break a bicolored cycle by recoloring one vertex in the cycle to a third color."""
    best_new_colors = None
    best_cost = count_violations_fast(colors)
    
    for v in cycle_vertices:
        old = colors[v]
        for new_c in range(k):
            if new_c == old:
                continue
            
            # Check if new_c is available (no neighbor has new_c)
            valid = True
            for u in adj[v]:
                if colors[u] == new_c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            colors[v] = new_c
            nc = count_violations_fast(colors)
            if nc < best_cost:
                best_cost = nc
                best_new_colors = list(colors)
            colors[v] = old
    
    if best_new_colors is not None:
        return best_new_colors
    return None

def local_search_with_cycle_breaking(k, initial=None, max_iter=10000, seed=None, deadline=None):
    """Local search that specifically targets and breaks bicolored cycles."""
    if seed is not None:
        random.seed(seed)
    
    if initial is None:
        colors = [random.randint(0, k-1) for _ in range(n)]
    else:
        colors = list(initial)
    
    current_cost = count_violations_fast(colors)
    best_cost = current_cost
    best_colors = list(colors)
    
    if current_cost == 0:
        return colors
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_cost == 0:
            return colors
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = list(colors)
        
        # If there are bicolored cycles, try to break them
        if current_cost > 0 and current_cost < 100:
            cycles = find_bicolored_cycles(colors)
            if cycles:
                # Pick a random cycle to break
                a, b, cycle_verts = random.choice(cycles)
                new_colors = break_cycle_recolor(colors, k, cycle_verts, a, b, deadline)
                if new_colors is not None:
                    colors = new_colors
                    current_cost = count_violations_fast(colors)
                    continue
        
        # Standard local search step
        v = random.randint(0, n-1)
        old = colors[v]
        
        best_new = old
        best_c = current_cost
        
        for new_c in range(k):
            if new_c == old:
                continue
            
            # Quick check: if any neighbor has new_c, skip
            if any(colors[u] == new_c for u in adj[v]):
                continue
            
            colors[v] = new_c
            nc = count_violations_fast(colors)
            if nc < best_c:
                best_c = nc
                best_new = new_c
            colors[v] = old
        
        if best_new != old:
            colors[v] = best_new
            current_cost = best_c
    
    if best_cost == 0:
        return best_colors
    return None

def structured_initialization(k):
    """Create a structured initial coloring based on graph pattern."""
    colors = [0] * n
    
    if k >= 3:
        # Try a pattern based on vertex index
        for i in range(n):
            colors[i] = i % k
    elif k == 2:
        for i in range(n):
            colors[i] = i % 2
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_result = None
    best_k = 50
    
    # First, try to find a 3-coloring using cycle-breaking local search
    for attempt in range(40):
        if time.time() > deadline - 20:
            break
        
        seed = 42 + attempt * 7
        init = structured_initialization(3)
        
        # Perturb the initialization
        for _ in range(5):
            v = random.randint(0, n-1)
            init[v] = random.randint(0, 2)
        
        result = local_search_with_cycle_breaking(3, initial=init, max_iter=5000, seed=seed, deadline=deadline - 15)
        if result is not None:
            if is_valid(result):
                best_result = result
                best_k = 3
                break
    
    # If no 3-coloring found, try 4-coloring
    if best_result is None:
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            seed = 100 + attempt * 13
            init = structured_initialization(4)
            
            for _ in range(3):
                v = random.randint(0, n-1)
                init[v] = random.randint(0, 3)
            
            result = local_search_with_cycle_breaking(4, initial=init, max_iter=8000, seed=seed, deadline=deadline - 5)
            if result is not None:
                if is_valid(result):
                    best_result = result
                    best_k = 4
                    break
    
    # If still no good result, try 5 colors
    if best_result is None:
        for attempt in range(10):
            if time.time() > deadline - 3:
                break
            
            seed = 300 + attempt * 19
            result = local_search_with_cycle_breaking(5, max_iter=5000, seed=seed, deadline=deadline - 1)
            if result is not None and is_valid(result):
                best_result = result
                best_k = 5
                break
    
    if best_result is None:
        # Last resort: use all distinct colors
        return list(range(n))
    
    return [int(c) for c in best_result]

# EVOLVE-BLOCK-END
