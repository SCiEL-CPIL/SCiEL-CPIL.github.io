"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 45.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_coloring(colors):
        """Check if a coloring is valid."""
        # Rule 1: adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Rule 2: for every pair of colors, induced subgraph is a forest
        color_to_verts = {}
        for i, c in enumerate(colors):
            if c not in color_to_verts:
                color_to_verts[c] = []
            color_to_verts[c].append(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_set = set(color_to_verts[c1]) | set(color_to_verts[c2])
                # Union-find to check forest
                parent = {v: v for v in verts_set}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for u, v in edges:
                    if u in verts_set and v in verts_set:
                        pu, pv = find(u), find(v)
                        if pu == pv:
                            return False
                        parent[pu] = pv
        return True
    
    # Try small k values with pattern-based approaches
    # The graph appears to have a grid-like structure with period 7 in one direction
    
    # Try k=3 with various patterns
    patterns_k3 = [
        lambda i: i % 3,
        lambda i: (i * 2) % 3,
        lambda i: (i // 7) % 3,
        lambda i: (i % 7) % 3,
        lambda i: ((i // 7) + (i % 7)) % 3,
        lambda i: ((i // 7) * 2 + (i % 7)) % 3,
    ]
    
    for pattern in patterns_k3:
        if time.time() > deadline:
            break
        colors = [pattern(i) for i in range(n)]
        if check_coloring(colors):
            return [int(c) for c in colors]
    
    # Try k=4 with various patterns
    patterns_k4 = [
        lambda i: i % 4,
        lambda i: (i * 2) % 4,
        lambda i: (i // 7) % 4,
        lambda i: (i % 7) % 4,
        lambda i: ((i // 7) + (i % 7)) % 4,
        lambda i: ((i // 7) * 2 + (i % 7)) % 4,
        lambda i: (i % 5) % 4,
    ]
    
    for pattern in patterns_k4:
        if time.time() > deadline:
            break
        colors = [pattern(i) for i in range(n)]
        if check_coloring(colors):
            return [int(c) for c in colors]
    
    # Try k=5 with backtracking and incremental bicolored constraint checking
    if time.time() < deadline - 10:
        k = 5
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        colors = [-1] * n
        
        # Precompute for faster bicolored check
        # Track which color pairs have been "used" in terms of edges
        # For incremental check, we need to detect if adding an edge creates a cycle in the bicolored subgraph
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            # Try each color, ordered by how constrained it is
            for c in range(k):
                # Check adjacency constraint
                ok = True
                for u in adj[v]:
                    if colors[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                colors[v] = c
                
                # Check bicolored constraint incrementally
                # For each neighbor u with color cu != c, check if edge (u,v) creates a cycle
                # in the subgraph induced by colors {c, cu}
                bicolored_ok = True
                for u in adj[v]:
                    cu = colors[u]
                    if cu == c:
                        continue
                    # Check if adding edge (u,v) to the {c,cu} subgraph creates a cycle
                    # We need to check if u and v are already connected in the {c,cu} subgraph
                    # without using edge (u,v)
                    # This requires a BFS/DFS in the {c,cu} subgraph, which is expensive
                    # For now, do a full check at the end
                    pass
                
                if bicolored_ok:
                    if backtrack(idx + 1):
                        return True
                
                colors[v] = -1
            return False
        
        if backtrack(0) and check_coloring(colors):
            return [int(c) for c in colors]
    
    # Try k=6 with backtracking
    if time.time() < deadline - 5:
        k = 6
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        colors = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            for c in range(k):
                ok = True
                for u in adj[v]:
                    if colors[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        if backtrack(0) and check_coloring(colors):
            return [int(c) for c in colors]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
