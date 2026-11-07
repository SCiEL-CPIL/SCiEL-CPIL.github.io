import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def is_valid(coloring):
    n = P['n']
    edges = P['edges']
    
    # Check proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
            
    # Check forest condition for all pairs of colors
    colors_used = set(coloring)
    
    if len(colors_used) < 2:
        return True
        
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = [i for i in range(n) if coloring[i] in (c1, c2)]
            v_set = set(verts)
            
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
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
                if u in v_set and v in v_set:
                    if not union(u, v):
                        return False
                        
    return True

def construct():
    n = P['n']
    
    # Try 4-coloring: even vertices get colors 0,1; odd vertices get colors 2,3
    # Pattern: color[i] = (i % 2) * 2 + (i // 2) % 2
    coloring4 = [(i % 2) * 2 + (i // 2) % 2 for i in range(n)]
    if is_valid(coloring4):
        return [int(c) for c in coloring4]
    
    # Try another 4-coloring pattern
    coloring4b = [(i % 2) * 2 + (i // 4) % 2 for i in range(n)]
    if is_valid(coloring4b):
        return [int(c) for c in coloring4b]
    
    # Try 5-coloring with modular approach
    for a in range(1, 6):
        for b in range(5):
            coloring5 = [(a * i + b) % 5 for i in range(n)]
            if is_valid(coloring5):
                return [int(c) for c in coloring5]
    
    # Fallback: return seed solution (always valid)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
