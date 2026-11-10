import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def has_bicolored_cycle(colors):
    """Check if any pair of colors induces a subgraph with a cycle (using union-find)."""
    # Group edges by color pair
    from collections import defaultdict
    color_pairs = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return True  # adjacent same color is invalid
        key = (min(cu, cv), max(cu, cv))
        color_pairs[key].append((u, v))
    
    for pair, edges in color_pairs.items():
        # Union-find to detect cycles
        parent = list(range(P['n']))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False  # cycle detected
            parent[px] = py
            return True
        for u, v in edges:
            if not union(u, v):
                return True
    return False

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    best_colors = None
    best_count = 50
    
    # Start with alternating 2-coloring
    base = [i % 2 for i in range(P['n'])]
    
    # Try adding 0, 1, 2, ... defects
    # A defect at position i changes base[i] to color 2
    # We need to find minimum number of defects to break all bicolored cycles
    
    # First, check if 2 colors work (no defects)
    if not has_bicolored_cycle(base):
        return [int(c) for c in base]
    
    # Try 1 defect
    for d in range(P['n']):
        candidate = base[:]
        candidate[d] = 2
        if not has_bicolored_cycle(candidate):
            return [int(c) for c in candidate]
    
    # Try 2 defects
    for d1 in range(P['n']):
        for d2 in range(d1 + 1, P['n']):
            if time.time() > deadline:
                break
            candidate = base[:]
            candidate[d1] = 2
            candidate[d2] = 2
            if not has_bicolored_cycle(candidate):
                return [int(c) for c in candidate]
        if time.time() > deadline:
            break
    
    # Try 3 defects
    for d1 in range(P['n']):
        for d2 in range(d1 + 1, P['n']):
            for d3 in range(d2 + 1, P['n']):
                if time.time() > deadline:
                    break
                candidate = base[:]
                candidate[d1] = 2
                candidate[d2] = 2
                candidate[d3] = 2
                if not has_bicolored_cycle(candidate):
                    return [int(c) for c in candidate]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Try 4 defects
    for d1 in range(P['n']):
        for d2 in range(d1 + 1, P['n']):
            for d3 in range(d2 + 1, P['n']):
                for d4 in range(d3 + 1, P['n']):
                    if time.time() > deadline:
                        break
                    candidate = base[:]
                    candidate[d1] = 2
                    candidate[d2] = 2
                    candidate[d3] = 2
                    candidate[d4] = 2
                    if not has_bicolored_cycle(candidate):
                        return [int(c) for c in candidate]
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Fallback: use 5 colors with a safe pattern
    # Pattern: i % 5
    candidate = [i % 5 for i in range(P['n'])]
    if not has_bicolored_cycle(candidate):
        return [int(c) for c in candidate]
    
    # Ultimate fallback: all distinct
    return list(range(P['n']))
