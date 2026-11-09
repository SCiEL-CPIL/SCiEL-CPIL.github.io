"""Minimize distinct colors for degree-4 circulant graph with bicolored-forest constraint.
Uses pattern-based construction with incremental union-find cycle detection."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _check_bicolored_forests(coloring):
    """Check that for every pair of colors present, the induced subgraph is a forest.
    Returns True if valid."""
    color_verts = {}
    for v in range(_N):
        c = coloring[v]
        if c < 0:
            return False
        if c not in color_verts:
            color_verts[c] = []
        color_verts[c].append(v)
    
    present = sorted(color_verts.keys())
    
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            # Union-find on vertices with color ci or cj
            parent = {}
            rank = {}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 0
                if y not in parent:
                    parent[y] = y
                    rank[y] = 0
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if (ca == ci and cb == cj) or (ca == cj and cb == ci):
                    if not union(a, b):
                        return False
    return True


def _is_valid(coloring):
    """Check proper coloring and bicolored-forest constraint."""
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    return _check_bicolored_forests(coloring)


def _try_pattern(k, pattern_fn, deadline):
    """Try a specific coloring pattern."""
    coloring = [pattern_fn(v) for v in range(_N)]
    if _is_valid(coloring):
        return coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try 2 colors (bipartite check)
    color = [-1] * _N
    for start in range(_N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            cu = color[u]
            for v in _ADJ[u]:
                if color[v] == -1:
                    color[v] = 1 - cu
                    q.append(v)
                elif color[v] == cu:
                    color = None
                    break
            if color is None:
                break
        if color is None:
            break
    
    if color is not None and _is_valid(color):
        return [int(x) for x in color]
    
    # Try 3 colors with various patterns
    if time.time() < deadline - 30:
        # Pattern: v mod 3
        c = [v % 3 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v + offset) mod 3
        for offset in range(3):
            c = [(v + offset) % 3 for v in range(_N)]
            if _is_valid(c):
                return [int(x) for x in c]
        
        # Pattern: (v // 2) % 3
        c = [(v // 2) % 3 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v * 2) % 3
        c = [(v * 2) % 3 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v * 3) % 3 = 0, not useful
        # Pattern: (v + v//10) % 3
        c = [(v + v // 10) % 3 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v * 7) % 3
        c = [(v * 7) % 3 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Try all permutations of 3-color assignments on small blocks
        # Pattern: (v % 5) mapped to 3 colors
        for perm in range(6):
            mapping = [0, 1, 2, 0, 1]
            # Apply permutation
            import itertools
            perms = list(itertools.permutations([0, 1, 2]))
            m = perms[perm]
            base = [0, 1, 2, 0, 1]
            c = [m[b] for b in base * 10]
            if len(c) == _N and _is_valid(c):
                return [int(x) for x in c]
    
    # Try 4 colors
    if time.time() < deadline - 20:
        # Pattern: v mod 4
        c = [v % 4 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v + offset) mod 4
        for offset in range(4):
            c = [(v + offset) % 4 for v in range(_N)]
            if _is_valid(c):
                return [int(x) for x in c]
        
        # Pattern: (v // 2) % 4
        c = [(v // 2) % 4 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v * 2) % 4
        c = [(v * 2) % 4 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v + v//10) % 4
        c = [(v + v // 10) % 4 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
        
        # Pattern: (v * 7) % 4
        c = [(v * 7) % 4 for v in range(_N)]
        if _is_valid(c):
            return [int(x) for x in c]
    
    # Try 5 colors (should always work as fallback)
    c = [v % 5 for v in range(_N)]
    if _is_valid(c):
        return [int(x) for x in c]
    
    # Fallback: greedy coloring
    coloring = [-1] * _N
    order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    for v in order:
        used = set()
        for nb in _ADJ[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        for c in range(10):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            coloring[v] = 9
    
    if _is_valid(coloring):
        return [int(x) for x in coloring]
    
    # Last resort: unique colors
    return [int(x) for x in range(_N)]
