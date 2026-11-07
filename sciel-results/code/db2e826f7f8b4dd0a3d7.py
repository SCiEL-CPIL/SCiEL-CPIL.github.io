"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolored_cycle(coloring, adj, c1, c2):
    """Check if subgraph induced by colors c1 and c2 has a cycle using union-find."""
    n = P['n']
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    
    for u in range(n):
        if coloring[u] in (c1, c2):
            for v in adj[u]:
                if v > u and coloring[v] in (c1, c2):
                    if not union(u, v):
                        return True
    return False

def _validate(coloring):
    """Check if coloring is valid: proper and no bicolored cycles."""
    n = P['n']
    adj = _build_adjacency()
    
    # Check proper coloring
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    
    # Check no bicolored cycles
    colors_used = set(coloring)
    color_list = list(colors_used)
    for i in range(len(color_list)):
        for j in range(i+1, len(color_list)):
            if _has_bicolored_cycle(coloring, adj, color_list[i], color_list[j]):
                return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def _greedy_color_with_forest_constraint(max_colors, adj, deadline):
    """Try to color with at most max_colors, ensuring bicolored forest property."""
    n = P['n']
    coloring = [-1] * n
    
    # Order vertices by degree (descending) for better greedy
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def _is_valid_assignment(coloring, v, c):
        # Check proper coloring
        for u in adj[v]:
            if coloring[u] == c:
                return False
        # Check bicolored forest constraint
        coloring[v] = c
        for c2 in set(coloring):
            if c2 != c:
                if _has_bicolored_cycle(coloring, adj, c, c2):
                    coloring[v] = -1
                    return False
        coloring[v] = -1
        return True
    
    # Try different color orderings
    for attempt in range(100):
        if time.time() > deadline:
            break
        random.shuffle(order)
        coloring = [-1] * n
        success = True
        for v in order:
            placed = False
            for c in range(max_colors):
                if _is_valid_assignment(coloring, v, c):
                    coloring[v] = c
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return coloring
    return None

def _local_search(initial_coloring, adj, deadline):
    """Improve coloring via local search."""
    n = P['n']
    current = list(initial_coloring)
    current_colors = _count_colors(current)
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        # Try to reduce colors
        for c in range(current_colors):
            # Try to recolor all vertices of color c to other colors
            vertices_of_c = [v for v in range(n) if current[v] == c]
            if not vertices_of_c:
                continue
            # Try to eliminate color c
            new_coloring = list(current)
            success = True
            for v in vertices_of_c:
                placed = False
                for nc in range(current_colors):
                    if nc == c:
                        continue
                    # Check if we can assign nc to v
                    valid = True
                    for u in adj[v]:
                        if new_coloring[u] == nc:
                            valid = False
                            break
                    if valid:
                        # Check bicolored forest
                        new_coloring[v] = nc
                        for c2 in set(new_coloring):
                            if c2 != nc:
                                if _has_bicolored_cycle(new_coloring, adj, nc, c2):
                                    valid = False
                                    break
                        if valid:
                            placed = True
                            break
                        else:
                            new_coloring[v] = -1
                if not placed:
                    success = False
                    break
            if success:
                # Verify
                if _validate(new_coloring):
                    current = new_coloring
                    current_colors = _count_colors(current)
    
    return current

def construct():
    deadline = time.time() + 85  # Safety margin
    
    adj = _build_adjacency()
    
    # Try from 3 colors upward
    best_coloring = None
    for max_c in range(3, 11):
        if time.time() > deadline:
            break
        coloring = _greedy_color_with_forest_constraint(max_c, adj, deadline)
        if coloring is not None:
            if best_coloring is None or _count_colors(coloring) < _count_colors(best_coloring):
                best_coloring = coloring
                if _count_colors(coloring) == max_c:
                    # Try to improve
                    improved = _local_search(coloring, adj, deadline)
                    if _validate(improved) and _count_colors(improved) <= _count_colors(best_coloring):
                        best_coloring = improved
                    break
    
    if best_coloring is None:
        # Fallback: use all distinct colors
        best_coloring = list(range(P['n']))
    
    # Ensure proper int type
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
