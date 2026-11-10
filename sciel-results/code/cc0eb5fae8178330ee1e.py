"""Structural coloring for C(50;{1,7}) with bicolored-forest constraint."""
import time
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

def _is_valid(colors, edges):
    """Full validity check: proper coloring + all bicolored subgraphs are forests."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    used = set(colors)
    for i, c1 in enumerate(used):
        for c2 in used:
            if c1 >= c2:
                continue
            if not _check_bicolored_forest(colors, edges, c1, c2):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Strategy 1: Try structural patterns based on the circulant structure
    # The graph C(50;{1,7}) has edges (i, i+1) and (i, i+7) mod 50
    # Try patterns that exploit the gcd structure
    
    # Pattern: (i // 25) * 2 + (i % 25) // 12  -> gives 4 colors
    for k in range(4, 25):
        if time.time() > deadline:
            break
        colors = [((i // 25) * 2 + (i % 25) // k) for i in range(n)]
        if _is_valid(colors, edges):
            return [int(c) for c in colors]
    
    # Pattern: (i % 25) // k + 2*(i // 25)
    for k in range(4, 25):
        if time.time() > deadline:
            break
        colors = [((i % 25) // k + 2 * (i // 25)) for i in range(n)]
        if _is_valid(colors, edges):
            return [int(c) for c in colors]
    
    # Pattern: (i * 7) % p for various p
    for p in range(3, 20):
        if time.time() > deadline:
            break
        colors = [(i * 7) % p for i in range(n)]
        if _is_valid(colors, edges):
            return [int(c) for c in colors]
    
    # Pattern: (i * a) % p for various a, p
    for p in range(3, 20):
        if time.time() > deadline:
            break
        for a in range(1, p):
            colors = [(i * a) % p for i in range(n)]
            if _is_valid(colors, edges):
                return [int(c) for c in colors]
    
    # Pattern: floor(i/p) mod q
    for p in range(2, 10):
        if time.time() > deadline:
            break
        for q in range(2, 10):
            colors = [(i // p) % q for i in range(n)]
            if _is_valid(colors, edges):
                return [int(c) for c in colors]
    
    # Pattern: (i % p) // q
    for p in range(3, 20):
        if time.time() > deadline:
            break
        for q in range(1, p):
            colors = [(i % p) // q for i in range(n)]
            if _is_valid(colors, edges):
                return [int(c) for c in colors]
    
    # Strategy 2: Backtracking with aggressive pruning
    # Try increasing number of colors from 3 upward
    for num_colors in range(3, 15):
        if time.time() > deadline:
            break
        
        colors = [None] * n
        # Process vertices in a smart order: start with vertex 0, then its neighbors
        # Use the natural order but with forward checking
        
        def bt(v_idx):
            if time.time() > deadline:
                return False
            if v_idx == n:
                return _is_valid(colors, edges)
            
            v = v_idx
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    forbidden.add(colors[nb])
            
            for c in range(num_colors):
                if c in forbidden:
                    continue
                
                colors[v] = c
                
                # Quick bicolored forest check for color c with each other used color
                valid = True
                used_colors = set()
                for i in range(v):
                    if colors[i] is not None:
                        used_colors.add(colors[i])
                for c2 in used_colors:
                    if c2 == c:
                        continue
                    if not _check_bicolored_forest(colors, edges, c, c2):
                        valid = False
                        break
                
                if valid:
                    if bt(v_idx + 1):
                        return True
                
                colors[v] = None
            
            return False
        
        if bt(0):
            return [int(c) for c in colors]
    
    # Fallback: use more colors
    for num_colors in range(15, 25):
        if time.time() > deadline:
            break
        
        colors = [None] * n
        
        def bt(v_idx):
            if time.time() > deadline:
                return False
            if v_idx == n:
                return _is_valid(colors, edges)
            
            v = v_idx
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    forbidden.add(colors[nb])
            
            for c in range(num_colors):
                if c in forbidden:
                    continue
                
                colors[v] = c
                
                valid = True
                used_colors = set()
                for i in range(v):
                    if colors[i] is not None:
                        used_colors.add(colors[i])
                for c2 in used_colors:
                    if c2 == c:
                        continue
                    if not _check_bicolored_forest(colors, edges, c, c2):
                        valid = False
                        break
                
                if valid:
                    if bt(v_idx + 1):
                        return True
                
                colors[v] = None
            
            return False
        
        if bt(0):
            return [int(c) for c in colors]
    
    # Last resort: unique colors
    return [int(i) for i in range(n)]
