"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolor_forest(colors, n, edges):
    """Check that for every pair of colors (a,b), the subgraph induced by vertices
    with color a or b is a forest (no cycles). Uses union-find per pair."""
    # Group vertices by color
    from collections import defaultdict
    color_vertices = defaultdict(list)
    for i in range(n):
        color_vertices[colors[i]].append(i)
    
    used_colors = sorted(color_vertices.keys())
    
    for idx_a in range(len(used_colors)):
        for idx_b in range(idx_a + 1, len(used_colors)):
            a, b = used_colors[idx_a], used_colors[idx_b]
            # Vertices with color a or b
            verts = set(color_vertices[a]) | set(color_vertices[b])
            # Build subgraph on these vertices
            parent = {v: v for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle
                parent[rx] = ry
                return True
            
            for u, v in edges:
                if u in verts and v in verts:
                    if not union(u, v):
                        return False  # found a bicolored cycle
    return True

def _backtrack_color(n, edges, k, deadline):
    """Backtracking search for a valid k-coloring with bicolored-forest constraint."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = [0] * n  # pos[v] = position in order
    for i, v in enumerate(order):
        pos[v] = i
    
    color = [-1] * n
    
    # Precompute for each vertex, which colors are forbidden by already-colored neighbors
    # We'll do this dynamically
    
    # Union-find structures for each color pair - but that's expensive.
    # Instead, we check at the end for small k, or use incremental checking.
    
    # For efficiency with small n=50, let's try a smarter approach:
    # Try all assignments with backtracking, checking constraints incrementally.
    
    # Incremental bicolored cycle detection:
    # When we assign color c to vertex v, for each neighbor u with color d != c,
    # we need to check that adding edge (v,u) to the subgraph of colors {c,d} doesn't create a cycle.
    
    # Maintain union-find for each color pair
    # With k small (say <= 8), number of pairs is at most 28.
    
    # For k colors, pairs are (i,j) for 0<=i<j<k
    # We'll maintain parent arrays for each pair
    
    import itertools
    
    def _valid_assignment(color, v, c):
        """Check if assigning color c to vertex v is valid given current partial coloring."""
        # Check adjacent vertices have different colors
        for u in adj[v]:
            if color[u] != -1 and color[u] == c:
                return False
        
        # Check bicolored cycle constraint
        # For each neighbor u with color d, the pair (c,d) subgraph must remain a forest
        # We need to verify that adding v (with color c) and its edges to neighbors with color d
        # doesn't create a cycle in the {c,d} subgraph.
        
        # Get all neighbors with their colors
        for u in adj[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            # Check pair (min(c,d), max(c,d))
            a, b = (c, d) if c < d else (d, c)
            # In the subgraph induced by colors {a, b}, adding vertex v (color a or b)
            # and edge (v, u) must not create a cycle.
            # This means: in the current {a,b} subgraph, v is not yet present (we're adding it).
            # So we just need to check that u is not already connected to v via other {a,b} paths.
            # But v is new, so the only way a cycle forms is if there's already a path from u to v
            # in the {a,b} subgraph excluding the edge we're adding. But v isn't in the subgraph yet!
            # So adding v with one edge to u cannot create a cycle by itself.
            # Wait - v might have OTHER neighbors with color a or b that are already colored.
            
        # Actually, v might have multiple neighbors with colors in {a,b}.
        # Let's collect all neighbors of v that have color a or b (excluding v itself since v is uncolored).
        
        for a, b in itertools.combinations(range(k), 2):
            if c not in (a, b):
                continue
            # v has color c, which is in pair (a,b)
            # Find all neighbors of v that have color a or b
            neighbors_in_pair = []
            for u in adj[v]:
                if color[u] in (a, b):
                    neighbors_in_pair.append(u)
            if len(neighbors_in_pair) <= 1:
                continue
            # More than one neighbor in this pair: check if they're all in the same connected component
            # of the current {a,b} subgraph. If two are in different components, adding v connects them
            # (no cycle). If two are in the same component, adding v creates a cycle.
            
            # Build the {a,b} subgraph from currently colored vertices
            # Vertices with color a or b
            verts_in_pair = set()
            for i in range(n):
                if color[i] in (a, b):
                    verts_in_pair.add(i)
            
            if len(verts_in_pair) < 3:
                continue
            
            # Union-find on verts_in_pair
            parent = {v_: v_ for v_ in verts_in_pair}
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
            
            for u, w in edges:
                if u in verts_in_pair and w in verts_in_pair:
                    union(u, w)
            
            # Now check if any two neighbors_in_pair are in the same component
            roots = set()
            for u in neighbors_in_pair:
                r = find(u)
                if r in roots:
                    return False
                roots.add(r)
        
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        for c in range(k):
            if _valid_assignment(color, v, c):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    if backtrack(0):
        return color
    return None

def construct():
    deadline = time.time() + 80  # 80s safety margin
    n = P['n']
    edges = P['edges']
    
    # Try increasing k
    for k in range(3, 13):
        if time.time() > deadline - 5:
            break
        result = _backtrack_color(n, edges, k, deadline)
        if result is not None:
            # Verify
            if _check_bicolor_forest(result, n, edges):
                return [int(x) for x in result]
    
    # Fallback: use a known valid coloring with more colors
    # 50 colors is always valid
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
