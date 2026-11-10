"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _valid(coloring, edges):
    # Proper coloring: adjacent vertices differ
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # No bicolored cycle: for each pair of colors, induced subgraph is a forest
    n = len(coloring)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Group vertices by color
    from collections import defaultdict
    color_to_verts = defaultdict(list)
    for i, c in enumerate(coloring):
        color_to_verts[c].append(i)
    colors = list(color_to_verts.keys())
    for idx, c1 in enumerate(colors):
        for c2 in colors[idx+1:]:
            # Check subgraph induced by vertices of color c1 or c2
            # Use union-find to detect cycle
            verts = color_to_verts[c1] + color_to_verts[c2]
            if len(verts) <= 1:
                continue
            # Build adjacency within this subset
            vset = set(verts)
            # Union-find
            parent = {v: v for v in verts}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                parent[rx] = ry
                return True
            # Count edges in induced subgraph; if edges >= vertices, cycle exists
            edge_count = 0
            for u in verts:
                for v in adj[u]:
                    if v in vset and u < v:
                        edge_count += 1
            if edge_count >= len(verts):
                return False
            # Also do union-find to be exact
            for u in verts:
                for v in adj[u]:
                    if v in vset and u < v:
                        if not union(u, v):
                            return False
    return True

def construct():
    edges = P['edges']
    n = P['n']
    # Precomputed valid 3-coloring
    # Pattern: i % 3 for most, with adjustments for wrap-around edges
    # Let's verify a systematic approach:
    # The graph has edges (i, i+1) for i=0..48, plus "diagonal" edges (i, i+7) for i=0..42
    # and wrap edges (0,49), (6,49), (42,49)
    # Try coloring: color[i] = i % 3
    # Check: (i, i+1): i%3 != (i+1)%3 ✓
    # Check: (i, i+7): i%3 vs (i+7)%3 = (i+1)%3 → different ✓
    # Check: (0,49): 0%3=0, 49%3=1 ✓
    # Check: (6,49): 6%3=0, 49%3=1 ✓
    # Check: (42,49): 42%3=0, 49%3=1 ✓
    # Now check bicolored cycles:
    # Colors 0,1,2 each appear ~17 times
    # For colors {0,1}: vertices where i%3 in {0,1}
    #   Edges between them: (i,i+1) where both in {0,1}: i%3=0→(i,i+1) both in set? i%3=0, (i+1)%3=1 yes
    #   i%3=1→(i,i+1): i%3=1, (i+1)%3=2 no
    #   So chain edges: 0-1, 3-4, 6-7, 9-10, ... (every 3rd pair starting at 0)
    #   Diagonal edges (i,i+7): i%3=0→(i+7)%3=1 yes; i%3=1→(i+7)%3=2 no
    #   So diagonal: 0-7, 3-10, 6-13, 9-16, ...
    #   This forms a graph where each vertex has degree up to 2 (one chain neighbor + one diagonal)
    #   Actually it's a collection of paths/cycles. Let's check if any cycle forms.
    #   Vertices with color 0: 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48
    #   Vertices with color 1: 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49
    #   Chain edges in {0,1}: (0,1),(3,4),(6,7),(9,10),(12,13),(15,16),(18,19),(21,22),(24,25),(27,28),(30,31),(33,34),(36,37),(39,40),(42,43),(45,46)
    #   Diagonal edges in {0,1}: (0,7),(3,10),(6,13),(9,16),(12,19),(15,22),(18,25),(21,28),(24,31),(27,34),(30,37),(33,40),(36,43),(39,46)
    #   Wait, (42,49): 42%3=0, 49%3=1 → yes this is also in the set
    #   And (45,49)? No, 45%3=0, 49%3=1, but is (45,49) an edge? No.
    #   Let me recount diagonal edges where i%3=0 and (i+7)%3=1: i in {0,3,6,9,12,15,18,21,24,27,30,33,36,39,42}
    #   That's 15 diagonal edges. Plus 16 chain edges. Total 31 edges on 34 vertices. 31 < 34, so forest. ✓
    # Similarly for other pairs. The i%3 coloring should work!
    coloring = [i % 3 for i in range(n)]
    # Verify
    if _valid(coloring, edges):
        return [int(c) for c in coloring]
    # Fallback: try other patterns
    # Try (i*2) % 5
    for mod in range(3, 15):
        coloring = [i % mod for i in range(n)]
        if _valid(coloring, edges):
            return [int(c) for c in coloring]
    # Last resort: unique colors
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
