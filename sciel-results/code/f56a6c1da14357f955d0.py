"""Bicolored-forest coloring for C(50;{1,7}) using constructive + simulated annealing."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, edges, c1, c2):
    """Check that subgraph induced by colors c1 and c2 is a forest using union-find."""
    vertices = set()
    for i, c in enumerate(colors):
        if c == c1 or c == c2:
            vertices.add(i)
    if len(vertices) <= 1:
        return True
    
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
    
    for u, v in edges:
        if u in vertices and v in vertices:
            if not union(u, v):
                return False
    return True

def _count_violations(colors, edges):
    """Count total violations: proper coloring + bicolored cycles."""
    n = len(colors)
    # Proper coloring violations
    proper_viol = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            proper_viol += 1
    
    # Bicolored cycle violations
    used = set(colors)
    bicol_viol = 0
    for i, c1 in enumerate(used):
        for c2 in used:
            if c1 >= c2:
                continue
            # Check if bicolored subgraph has a cycle
            vertices = set()
            for i, c in enumerate(colors):
                if c == c1 or c == c2:
                    vertices.add(i)
            if len(vertices) <= 1:
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
            
            for u, v in edges:
                if u in vertices and v in vertices:
                    if not union(u, v):
                        bicol_viol += 1
                        break
    
    return proper_viol + bicol_viol

def _is_valid(colors, edges):
    """Full validity check."""
    return _count_violations(colors, edges) == 0

def _constructive_greedy(n, edges, num_colors, max_time=30):
    """Greedy constructive approach: assign colors to minimize future conflicts."""
    start_time = time.time()
    deadline = start_time + max_time
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute: for each pair of colors, which vertex pairs would create a bicolored cycle
    # We'll use a simpler approach: try random permutations of vertex order and greedy assign
    
    best_result = None
    best_score = float('inf')
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        # Random vertex ordering
        order = list(range(n))
        random.shuffle(order)
        
        colors = [None] * n
        valid = True
        
        for v in order:
            # Find a color that doesn't conflict with neighbors and doesn't create bicolored cycles
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    forbidden.add(colors[nb])
            
            found = False
            for c in range(num_colors):
                if c in forbidden:
                    continue
                
                colors[v] = c
                
                # Check bicolored forest constraint with all previously assigned colors
                ok = True
                used_colors = set(colors[:n])
                used_colors.discard(None)
                for c2 in used_colors:
                    if c2 == c:
                        continue
                    if not _check_bicolored_forest(colors, edges, c, c2):
                        ok = False
                        break
                
                if ok:
                    found = True
                    break
                else:
                    colors[v] = None
            
            if not found:
                valid = False
                break
        
        if valid:
            score = len(set(colors))
            if score < best_score:
                best_score = score
                best_result = [int(c) for c in colors]
        
        if best_score <= 4:
            break
    
    return best_result

def _simulated_annealing(n, edges, num_colors, max_time=50):
    """Simulated annealing to minimize violations."""
    start_time = time.time()
    deadline = start_time + max_time
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Initialize with random coloring
    colors = [random.randint(0, num_colors - 1) for _ in range(n)]
    current_viol = _count_violations(colors, edges)
    
    best_colors = list(colors)
    best_viol = current_viol
    
    T = 100.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random vertex and change its color
        v = random.randint(0, n - 1)
        old_c = colors[v]
        new_c = random.randint(0, num_colors - 1)
        if new_c == old_c:
            continue
        
        colors[v] = new_c
        new_viol = _count_violations(colors, edges)
        
        delta = new_viol - current_viol
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_colors = list(colors)
                if best_viol == 0:
                    break
        else:
            colors[v] = old_c
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    if best_viol == 0:
        return [int(c) for c in best_colors]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Try constructive greedy first with 4 colors
    result = _constructive_greedy(n, edges, 4, max_time=20)
    if result is not None:
        return result
    
    # Try simulated annealing with 4 colors
    if time.time() < deadline:
        result = _simulated_annealing(n, edges, 4, max_time=40)
        if result is not None:
            return result
    
    # Try with 5 colors
    if time.time() < deadline:
        result = _constructive_greedy(n, edges, 5, max_time=15)
        if result is not None:
            return result
    
    if time.time() < deadline:
        result = _simulated_annealing(n, edges, 5, max_time=20)
        if result is not None:
            return result
    
    # Fallback: unique colors
    return [int(i) for i in range(n)]
