"""Solver for few-color graph coloring with no bicolored cycles."""
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']

def _build_adj():
    adj = [[] for _ in range(_N)]
    for u, v in _EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_no_bicolored_cycles(coloring, adj):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = _N
    # Group vertices by color
    by_color = defaultdict(list)
    for i in range(n):
        by_color[coloring[i]].append(i)
    
    color_list = list(by_color.keys())
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            c1, c2 = color_list[i], color_list[j]
            verts = set(by_color[c1]) | set(by_color[c2])
            # Check if induced subgraph on verts is a forest
            # Use union-find
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False  # cycle detected
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                return True
            
            for v in verts:
                for w in adj[v]:
                    if w in verts and coloring[w] in (c1, c2):
                        if not union(v, w):
                            return False
    return True

def _solve_max_colors(max_colors, adj, order):
    """Try to color the graph with at most max_colors, avoiding bicolored cycles."""
    n = _N
    coloring = [-1] * n
    # Precompute for each color pair, which vertices have that color
    # We'll track edges within each color pair incrementally
    
    # For incremental checking, track for each (c1,c2) pair:
    # - set of vertices with color c1 or c2
    # - union-find state
    # This is complex, so let's do a simpler approach:
    # After assigning each vertex, check locally if any bicolored cycle forms
    
    def _is_valid_assignment(v, c, coloring):
        # Check: for each neighbor u with color cu, and for each other neighbor w with color cw,
        # if cu != cw and cu != c and cw != c, we need to check if adding edge v-w (through color c)
        # creates a cycle in the subgraph induced by colors {cu, cw}
        # Actually simpler: check all pairs (c, cu) for neighbors u with color cu
        for u in adj[v]:
            cu = coloring[u]
            if cu == -1 or cu == c:
                continue
            # Now check if in the subgraph induced by colors {c, cu}, adding v (color c) connected to u (color cu)
            # creates a cycle. This means: is there already a path from v to u using only vertices
            # colored c or cu (excluding the direct edge v-u)?
            # BFS from v using only vertices colored c or cu, not using edge to u directly
            # Actually: BFS from u using only vertices colored c or cu, see if we reach v without using edge (v,u)
            # Simpler: BFS from v, but don't go through u. If we reach u, cycle.
            # Wait, we're adding v with color c. The edges from v go to neighbors with color cu.
            # We need to check if there's a path from v to u in the existing graph (before adding v's edges)
            # using only vertices colored c or cu. But v is new, so the path would be: v -> (neighbor with color c or cu) -> ... -> u
            # Actually the existing graph already has all vertices except v assigned.
            # So: in the subgraph induced by {c, cu} colors on vertices excluding v, is there a path from any neighbor of v (with color c or cu) to u?
            # More precisely: v connects to all its neighbors. For the pair (c, cu), v connects to neighbors with color cu (and neighbors with color c, but those don't form bicolored edges with cu).
            # The bicolored edges for pair (c,cu) involving v are: (v, u') for each neighbor u' with color cu.
            # A cycle forms if there's already a path between two of these neighbors in the subgraph induced by {c, cu}.
            # Or: a path from one such neighbor to u in the {c,cu} subgraph.
            
            # Let's check: among neighbors of v with color cu, is there a pair that are connected in the {c,cu} induced subgraph (excluding v)?
            # If yes, then adding v creates a cycle.
            # Also: is there a path from u to any other neighbor of v with color cu in the {c,cu} subgraph? That would also create a cycle.
            
            # Simplification: BFS from u in the {c,cu} induced subgraph (excluding v). If we reach any other neighbor of v with color cu, cycle.
            # Also if we reach a neighbor of v with color c... wait, color c vertices are also in the subgraph.
            # The subgraph induced by colors {c, cu} includes all vertices colored c or cu.
            # v is being added with color c. Its neighbors in this subgraph are: neighbors with color c and neighbors with color cu.
            # Edges in the subgraph: edges between vertices both colored c or both colored cu or one c and one cu.
            # When we add v (color c), new edges are (v, w) for w in adj[v] where coloring[w] in {c, cu}.
            # A cycle is created if there's already a path from v to some neighbor w in the subgraph (excluding the new edges from v).
            # Since v is new, paths from v must go through its neighbors. So: is there a path between two neighbors of v in the {c,cu} subgraph (excluding v)?
            # If yes, cycle.
            
            # For efficiency, let's just check: for the pair (c, cu), collect all neighbors of v with color in {c, cu}.
            # Then check if any two of them are connected in the {c,cu} induced subgraph (on vertices != v).
            
            # This is getting expensive. Let's use a different approach for small max_colors.
            pass
        
        # Alternative: just do full check after each assignment (slow but correct)
        # For 50 vertices and small color counts, this might be OK with pruning.
        return True
    
    # Given complexity, let's use a cleaner approach:
    # Try all colorings with max_colors via backtracking, with pruning.
    # Pruning: after each assignment, check if any bicolored cycle exists.
    
    sys.setrecursionlimit(10000)
    
    def _has_bicolored_cycle_partial(coloring, assigned_count):
        """Check if current partial coloring has a bicolored cycle among assigned vertices."""
        # Only check among assigned vertices
        assigned = set(i for i in range(n) if coloring[i] != -1)
        by_color = defaultdict(list)
        for i in assigned:
            by_color[coloring[i]].append(i)
        
        color_list = list(by_color.keys())
        for i in range(len(color_list)):
            for j in range(i + 1, len(color_list)):
                c1, c2 = color_list[i], color_list[j]
                verts = set(by_color[c1]) | set(by_color[c2])
                if len(verts) < 3:
                    continue
                # Union-find
                parent = {v: v for v in verts}
                rank = {v: 0 for v in verts}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra == rb:
                        return False
                    if rank[ra] < rank[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    if rank[ra] == rank[rb]:
                        rank[ra] += 1
                    return True
                
                for v in verts:
                    for w in adj[v]:
                        if w in verts and v < w:  # avoid double counting
                            if not union(v, w):
                                return True
        return False
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Try each color
        for c in range(max_colors):
            # Check adjacency constraint
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            coloring[v] = c
            # Check bicolored cycles
            if _has_bicolored_cycle_partial(coloring, idx + 1):
                coloring[v] = -1
                continue
            if backtrack(idx + 1):
                return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring
    return None

def _get_deg_order(adj):
    """Get vertices in degeneracy order (lowest degree first)."""
    n = _N
    remaining = set(range(n))
    order = []
    while remaining:
        # Find vertex with minimum degree in remaining subgraph
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = sum(1 for u in adj[v] if u in remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        order.append(min_v)
        remaining.remove(min_v)
    return order

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    adj = _build_adj()
    order = _get_deg_order(adj)
    
    # Try increasing number of colors
    best_solution = list(range(_N))  # Fallback: 50 colors always works
    best_colors = _N
    
    for max_c in range(3, _N + 1):
        if time.time() > deadline - 5:
            break
        result = _solve_max_colors(max_c, adj, order)
        if result is not None:
            # Verify
            if _check_no_bicolored_cycles(result, adj):
                # Check adjacency
                ok = True
                for u, v in _EDGES:
                    if result[u] == result[v]:
                        ok = False
                        break
                if ok:
                    best_solution = result
                    best_colors = max_c
                    break
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
