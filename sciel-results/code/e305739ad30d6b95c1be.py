"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
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
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _solve_with_rollback(n, edges, k, deadline):
    """
    Backtracking solver with incremental union-find (rollback) for
    bicolored forest constraint checking.
    """
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map pair (c1,c2) with c1<c2 to an index
    pair_to_idx = {}
    all_pairs = []
    for i in range(k):
        for j in range(i+1, k):
            pair_to_idx[(i, j)] = len(all_pairs)
            all_pairs.append((i, j))
    num_pairs = len(all_pairs)
    
    # For each pair, maintain a union-find structure with rollback
    # parent[pair_idx][v] = parent of v in the DSU for that pair
    # size[pair_idx][v] = size of component (for union by size)
    # history stack for rollback
    dsu_parent = [[v for v in range(n)] for _ in range(num_pairs)]
    dsu_size = [[1] * n for _ in range(num_pairs)]
    # history: list of (pair_idx, x, y, parent_y, size_x) for rollback
    history = []
    
    def dsu_find(pi, x):
        while dsu_parent[pi][x] != x:
            dsu_parent[pi][x] = dsu_parent[pi][dsu_parent[pi][x]]
            x = dsu_parent[pi][x]
        return x
    
    def dsu_union(pi, x, y):
        """Union x and y in DSU for pair pi. Returns True if merged, False if same component."""
        rx = dsu_find(pi, x)
        ry = dsu_find(pi, y)
        if rx == ry:
            return False
        # Union by size
        if dsu_size[pi][rx] < dsu_size[pi][ry]:
            rx, ry = ry, rx
        # Record for rollback
        history.append((pi, rx, ry, dsu_parent[pi][ry], dsu_size[pi][rx]))
        dsu_parent[pi][ry] = rx
        dsu_size[pi][rx] += dsu_size[pi][ry]
        return True
    
    def rollback_to(level):
        """Rollback DSU state to the given history level."""
        while len(history) > level:
            pi, rx, ry, old_parent_ry, old_size_rx = history.pop()
            dsu_parent[pi][ry] = old_parent_ry
            dsu_size[pi][rx] = old_size_rx
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
        
        v = order[idx]
        
        # Get forbidden colors from colored neighbors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        # Try each color
        for color in range(k):
            if color in forbidden:
                continue
            
            colors[v] = color
            level = len(history)
            
            # Check affected pairs and add edges to DSUs
            valid = True
            for nb in adj[v]:
                if colors[nb] == -1:
                    continue
                nb_color = colors[nb]
                if nb_color == color:
                    # Shouldn't happen due to forbidden check, but safety
                    valid = False
                    break
                c1 = min(color, nb_color)
                c2 = max(color, nb_color)
                pi = pair_to_idx[(c1, c2)]
                if not dsu_union(pi, v, nb):
                    # Cycle detected in this bicolored subgraph
                    valid = False
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            colors[v] = -1
            rollback_to(level)
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _solve_k_coloring(n, edges, k, deadline, rng):
    """General k-coloring solver with backtracking using rollback DSU."""
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # All possible color pairs
    pair_to_idx = {}
    all_pairs = []
    for i in range(k):
        for j in range(i+1, k):
            pair_to_idx[(i, j)] = len(all_pairs)
            all_pairs.append((i, j))
    num_pairs = len(all_pairs)
    
    dsu_parent = [[v for v in range(n)] for _ in range(num_pairs)]
    dsu_size = [[1] * n for _ in range(num_pairs)]
    history = []
    
    def dsu_find(pi, x):
        while dsu_parent[pi][x] != x:
            dsu_parent[pi][x] = dsu_parent[pi][dsu_parent[pi][x]]
            x = dsu_parent[pi][x]
        return x
    
    def dsu_union(pi, x, y):
        rx = dsu_find(pi, x)
        ry = dsu_find(pi, y)
        if rx == ry:
            return False
        if dsu_size[pi][rx] < dsu_size[pi][ry]:
            rx, ry = ry, rx
        history.append((pi, rx, ry, dsu_parent[pi][ry], dsu_size[pi][rx]))
        dsu_parent[pi][ry] = rx
        dsu_size[pi][rx] += dsu_size[pi][ry]
        return True
    
    def rollback_to(level):
        while len(history) > level:
            pi, rx, ry, old_parent_ry, old_size_rx = history.pop()
            dsu_parent[pi][ry] = old_parent_ry
            dsu_size[pi][rx] = old_size_rx
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
        
        v = order[idx]
        
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        # Try colors in a deterministic order first, then shuffled
        color_order = list(range(k))
        rng.shuffle(color_order)
        
        for color in color_order:
            if color in forbidden:
                continue
            
            colors[v] = color
            level = len(history)
            
            valid = True
            for nb in adj[v]:
                if colors[nb] == -1:
                    continue
                nb_color = colors[nb]
                if nb_color == color:
                    valid = False
                    break
                c1 = min(color, nb_color)
                c2 = max(color, nb_color)
                pi = pair_to_idx[(c1, c2)]
                if not dsu_union(pi, v, nb):
                    valid = False
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            colors[v] = -1
            rollback_to(level)
        
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 80  # Safety margin (evaluator kills at 90s)
    
    # Try 3 colors first with rollback DSU solver
    result = _solve_with_rollback(n, edges, 3, deadline)
    if result is not None:
        return [int(c) for c in result]
    
    # Try 4 colors
    rng = random.Random(42)
    result = _solve_k_coloring(n, edges, 4, deadline, rng)
    if result is not None:
        return [int(c) for c in result]
    
    # Try 5 colors
    result = _solve_k_coloring(n, edges, 5, deadline, rng)
    if result is not None:
        return [int(c) for c in result]
    
    # Fallback: greedy assignment with all different colors
    return list(range(n))
# EVOLVE-BLOCK-END
