"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

import time
import random

def _has_bicolored_cycle(adj, color, n, c1, c2):
    # Build subgraph induced by vertices of color c1 or c2; check if it's a forest
    # Using union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # cycle
        parent[ra] = rb
        return True
    for u in range(n):
        if color[u] == c1:
            for v in adj[u]:
                if v > u and color[v] == c2:
                    if not union(u, v):
                        return True
    return False

def _is_valid(color, adj, n):
    # Check adjacent vertices differ
    for u in range(n):
        for v in adj[u]:
            if v > u and color[u] == color[v]:
                return False
    # Check each color pair forms a forest
    used = set(color)
    clist = sorted(used)
    for i in range(len(clist)):
        for j in range(i+1, len(clist)):
            if _has_bicolored_cycle(adj, color, n, clist[i], clist[j]):
                return False
    return True

def _count_colors(color):
    return len(set(color))

def _greedy_color(adj, n, order, max_colors):
    # Greedy coloring with given order and max colors
    color = [-1] * n
    for u in order:
        used = set()
        for v in adj[u]:
            if color[v] != -1:
                used.add(color[v])
        for c in range(max_colors):
            if c not in used:
                color[u] = c
                break
        else:
            return None
    return color

def _try_fix(color, adj, n, max_colors, iters=20000):
    # Local search to eliminate bicolored cycles
    best = color[:]
    best_score = _count_colors(best)
    for it in range(iters):
        if not _is_valid(best, adj, n):
            # Find a problematic pair
            clist = sorted(set(best))
            fixed = False
            for i in range(len(clist)):
                for j in range(i+1, len(clist)):
                    if _has_bicolored_cycle(adj, best, n, clist[i], clist[j]):
                        # Pick a random vertex of one of these colors and recolor
                        verts = [u for u in range(n) if best[u] in (clist[i], clist[j])]
                        u = random.choice(verts)
                        # Try to recolor to another color
                        used = set()
                        for v in adj[u]:
                            used.add(best[v])
                        for c in range(max_colors):
                            if c not in used and c != best[u]:
                                best[u] = c
                                fixed = True
                                break
                        if not fixed:
                            # Try increasing color space
                            c = max_colors
                            for v in adj[u]:
                                if best[v] == c:
                                    c += 1
                            best[u] = c
                        break
                if fixed:
                    break
            if not fixed:
                break
        else:
            # Valid! Try to reduce colors
            break
    return best

def construct():
    deadline = time.time() + 50
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # Try increasing number of colors
    for max_colors in range(2, 50):
        if time.time() > deadline - 5:
            break
        # Multiple random orders
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            order = list(range(n))
            random.shuffle(order)
            color = _greedy_color(adj, n, order, max_colors)
            if color is None:
                continue
            if _is_valid(color, adj, n):
                # Found valid coloring with max_colors
                # Try to reduce further
                return [int(c) for c in color]
            # Try local search
            fixed = _try_fix(color, adj, n, max_colors, iters=5000)
            if _is_valid(fixed, adj, n):
                return [int(c) for c in fixed]
    
    # Fallback: use many colors (always valid)
    return list(range(n))
# EVOLVE-BLOCK-END

