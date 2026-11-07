"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # For tracking bicolored forest constraint:
    # For each pair of colors (c1, c2), we need the subgraph induced by vertices
    # colored c1 or c2 to be a forest. We use union-find per color pair.
    
    # We'll do backtracking: assign colors one vertex at a time in the order.
    # For each pair of colors used, maintain a union-find structure.
    
    # Actually, let's think differently. The constraint is: for any two colors c1, c2,
    # the edges between vertices colored c1 and c2 (plus edges within c1 and within c2... 
    # wait, no. The subgraph induced by colors {c1, c2} means all vertices with color c1 or c2,
    # and all edges among them. This must be a forest.
    
    # So for each pair of colors, the bipartite-like graph (actually it includes edges within same color too)
    # must be acyclic.
    
    # Let me use a different approach: try small numbers of colors first.
    # Start with k=3, 4, 5, ... and try to find a valid coloring.
    
    # For a given k, use backtracking with constraint propagation.
    
    deadline = time.time() + 55  # leave margin
    
    # Try increasing number of colors
    for k in range(3, 50):
        if time.time() > deadline - 5:
            break
        result = _solve_with_k_colors(n, adj, order, k, deadline)
        if result is not None:
            return result
    
    # Fallback: return a valid coloring with more colors
    # Use greedy with k=10
    result = _solve_with_k_colors(n, adj, order, 10, deadline)
    if result is not None:
        return result
    
    # Last resort: trivial coloring
    return list(range(n))


def _solve_with_k_colors(n, adj, order, k, deadline):
    """Try to color graph with exactly k colors (0..k-1) satisfying constraints."""
    color = [-1] * n  # -1 means unassigned
    
    # For each pair of colors (i, j), maintain union-find to detect cycles
    # in the subgraph induced by colors {i, j}
    
    # Union-find structures: uf[(i,j)] where i < j
    # Each union-find has parent array of size n
    
    from collections import defaultdict as dd
    uf_parents = {}  # (i,j) -> list of parents
    
    def get_uf(ci, cj):
        key = (min(ci, cj), max(ci, cj))
        if key not in uf_parents:
            uf_parents[key] = list(range(n))
        return uf_parents[key]
    
    def uf_find(parents, x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    
    def uf_union(parents, x, y):
        rx = uf_find(parents, x)
        ry = uf_find(parents, y)
        if rx == ry:
            return False  # cycle detected
        parents[rx] = ry
        return True
    
    # Track which edges have been "processed" for each color pair
    # When we assign a color to a vertex, we need to union it with neighbors
    # that already have a color in the same pair
    
    # For efficiency, track for each color pair, which vertices are "active"
    
    # Actually, let's just do straightforward backtracking with incremental union-find.
    # When assigning color c to vertex v:
    #   For each neighbor u of v:
    #     If color[u] != -1:
    #       Let cu = color[u]
    #       If cu == c: 
    #         # Edge (v,u) is within color c. 
    #         # This edge is in the subgraph for pair (c, c') for any c'... 
    #         # Wait no. The subgraph induced by colors {c, c'} includes edges
    #         # where both endpoints have color in {c, c'}.
    #         # An edge (v,u) where both have color c is in the subgraph for pair (c, c')
    #         # for any c'. So we need to check all pairs involving c.
    #         # Hmm, this is getting complex.
    #       
    #       Let me reconsider. The constraint is: for each pair {c1, c2}, the subgraph
    #       induced by vertices with color in {c1, c2} must be a forest.
    #       
    #       When we add vertex v with color c, we add edges from v to its neighbors.
    #       For each neighbor u with color cu:
    #         The edge (v, u) is part of the subgraph for pair {c, cu} (if c != cu)
    #         or pair {c, c} which doesn't make sense... 
    #         Actually, if c == cu, the edge is within color c, and it's part of
    #         the subgraph for any pair {c, c'} where c' != c.
    #         
    #         Wait, I think I'm overcomplicating. Let me re-read the constraint.
    #         
    #         "the subgraph induced by any two colors must be a forest"
    #         
    #         So for each unordered pair of distinct colors {c1, c2}, consider the set of
    #         vertices with color c1 or c2, and the edges among them. This must be a forest.
    #         
    #         An edge (v, u) where v has color c1 and u has color c2 is in the subgraph
    #         for pair {c1, c2}.
    #         An edge (v, u) where both have color c1 is in the subgraph for pair {c1, c2}
    #         for any c2 != c1.
    #         
    #         So when we assign color c to v, and v has neighbor u with color cu:
    #         - If cu != c: the edge (v,u) is in subgraph for pair {c, cu}. We need to
    #           check if adding this edge creates a cycle in that subgraph.
    #         - If cu == c: the edge (v,u) is in subgraph for pair {c, c'} for all c' != c.
    #           We need to check all such pairs.
    
    # This is getting complex. Let me use a simpler but still effective approach:
    # After each assignment, check all affected color pairs for cycles.
    
    # Actually, let me use a different strategy: just check the full constraint after
    # each assignment by building the relevant subgraphs. For 50 vertices, this is fast enough.
    
    # Let me use a cleaner backtracking:
    
    def is_valid_assignment(color, v, c):
        """Check if assigning color c to vertex v maintains all constraints."""
        # Check adjacent vertices have different colors
        for u in adj[v]:
            if color[u] == c:
                return False
        
        # Check bicolored forest constraint
        # For each pair of colors {c, cu} where cu is a color used by a neighbor of v
        # or cu == c, check if adding v with color c creates a cycle
        
        # Get all colors that appear among v's neighbors (and c itself)
        relevant_colors = set()
        relevant_colors.add(c)
        for u in adj[v]:
            if color[u] != -1:
                relevant_colors.add(color[u])
        
        # For each pair in relevant_colors, check if the subgraph is a forest
        # We only need to check pairs that involve c (since other pairs were already valid)
        for cu in relevant_colors:
            if cu == c:
                continue
            # Check pair {c, cu}
            # Build the subgraph induced by vertices with color c or cu
            # and check if it's a forest
            if not _is_forest(n, adj, color, c, cu):
                return False
        
        # Also, if c appears in multiple places, edges within color c are in pairs {c, c'}
        # for all c'. But we already checked pairs involving c above.
        # However, we also need to check that edges within color c don't create cycles
        # in pairs {c, c'} where c' is a color NOT in relevant_colors but c' is used elsewhere.
        # Hmm, this is the issue. If v has color c, and there's another vertex w with color c,
        # and there's a path between them through color c vertices, that's fine. But if
        # adding v creates a cycle in the color-c subgraph, that cycle would be in the
        # subgraph for pair {c, c'} for any c'.
        
        # So we need to also check: is the subgraph induced by color c alone a forest?
        # Actually no - the constraint is on pairs of DISTINCT colors. A single color's
        # induced subgraph being a forest is not directly required, but if it has a cycle,
        # then for any other color c', the pair {c, c'} subgraph would contain that cycle.
        
        # So: the subgraph induced by color c must be a forest (otherwise any pair {c, c'} 
        # would have a cycle). Similarly for cu.
        
        # Let me just check: for the pair {c, cu}, is the induced subgraph a forest?
        # This covers all cases.
        
        return True
    
    def _is_forest(n, adj, color, c1, c2):
        """Check if subgraph induced by vertices with color c1 or c2 is a forest."""
        # Collect vertices with color c1 or c2
        verts = set()
        for v in range(n):
            if color[v] == c1 or color[v] == c2:
                verts.add(v)
        
        # Build adjacency within this subgraph
        # Use union-find to check for cycles
        parent = {v: v for v in verts}
        
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
        
        for v in verts:
            for u in adj[v]:
                if u in verts and u > v:  # avoid double counting
                    if not union(v, u):
                        return False
        
        return True
    
    # Backtracking
    def backtrack(idx):
        if time.time() > deadline - 2:
            return None
        
        if idx == n:
            return list(color)
        
        v = order[idx]
        
        # Try colors 0 to k-1
        for c in range(k):
            # Check if color c is allowed for v
            ok = True
            for u in adj[v]:
                if color[u] == c:
                    ok = False
                    break
            
            if not ok:
                continue
            
            # Tentatively assign
            color[v] = c
            
            # Check bicolored forest constraints
            valid = True
            relevant_colors = set([c])
            for u in adj[v]:
                if color[u] != -1:
                    relevant_colors.add(color[u])
            
            for cu in relevant_colors:
                if cu == c:
                    continue
                if not _is_forest(n, adj, color, c, cu):
                    valid = False
                    break
            
            if not valid:
                color[v] = -1
                continue
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            color[v] = -1
        
        return None
    
    return backtrack(0)


# EVOLVE-BLOCK-END
