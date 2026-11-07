"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest_full(colors):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = len(colors)
    color_to_verts = defaultdict(list)
    for i, c in enumerate(colors):
        color_to_verts[c].append(i)
    
    adj = _build_adj()
    colors_used = set(colors)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = set(color_to_verts[c1] + color_to_verts[c2])
            if len(verts) <= 1:
                continue
            parent = {v: v for v in verts}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        if not union(u, v):
                            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    adj = _build_adj()
    
    # Compute degeneracy order
    deg = {v: len(adj[v]) for v in range(n)}
    remaining = set(range(n))
    order = []
    while remaining:
        min_deg = min(deg[v] for v in remaining)
        candidates = [v for v in remaining if deg[v] == min_deg]
        v = min(candidates)
        order.append(v)
        remaining.remove(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    def try_coloring_v2(k, order_seq=None):
        if order_seq is None:
            order_seq = order
        colors = [-1] * n
        pos_in_order_local = {v: i for i, v in enumerate(order_seq)}
        
        def check_assignment(v, c, colors_so_far, idx):
            """Check if assigning color c to vertex v is valid given colors_so_far."""
            other_colors = set()
            for u in adj[v]:
                u_pos = pos_in_order_local[u]
                if u_pos < idx:
                    other_colors.add(colors[u_pos])
            
            for c2 in other_colors:
                if c2 == c:
                    continue
                # Build the {c, c2} subgraph including v
                verts = set()
                verts.add(v)
                for i in range(idx):
                    if colors[i] == c or colors[i] == c2:
                        verts.add(order_seq[i])
                
                if len(verts) <= 1:
                    continue
                
                # Union-find on this subgraph
                parent = {v: v for v in verts}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    parent[px] = py
                    return True
                
                for u in verts:
                    for w in adj[u]:
                        if w in verts and u < w:
                            if not union(u, w):
                                return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return colors[:]
            
            v = order_seq[idx]
            # Get colors of already-colored neighbors
            neighbor_colors = set()
            for u in adj[v]:
                u_pos = pos_in_order_local[u]
                if u_pos < idx:
                    neighbor_colors.add(colors[u_pos])
            
            for c in range(k):
                if c in neighbor_colors:
                    continue
                if check_assignment(v, c, colors, idx):
                    colors[idx] = c
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    colors[idx] = -1
            return None
        
        return backtrack(0)
    
    for k in range(2, 12):
        if time.time() > deadline:
            break
        result = try_coloring_v2(k)
        if result is not None:
            final_colors = [0] * n
            for i, v in enumerate(order):
                final_colors[v] = result[i]
            # Verify
            assert _check_bicolored_forest_full(final_colors)
            for u, v in P['edges']:
                assert final_colors[u] != final_colors[v]
            color_map = {}
            remapped = []
            for c in final_colors:
                if c not in color_map:
                    color_map[c] = len(color_map)
                remapped.append(color_map[c])
            return [int(x) for x in remapped]
    
    # Fallback: greedy with many colors
    final_colors = [0] * n
    for v in order:
        neighbor_colors = set()
        for u in adj[v]:
            if final_colors[u] != 0 or u != v:
                if final_colors[u] != -1:
                    neighbor_colors.add(final_colors[u])
        c = 0
        while c in neighbor_colors:
            c += 1
        final_colors[v] = c
    
    color_map = {}
    remapped = []
    for c in final_colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        remapped.append(color_map[c])
    return [int(x) for x in remapped]
# EVOLVE-BLOCK-END
