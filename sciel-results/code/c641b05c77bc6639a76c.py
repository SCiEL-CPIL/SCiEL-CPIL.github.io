"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

# Order vertices by degree (all degree 4 here) then by index for determinism
# Use a smarter ordering: process vertices in an order that maximizes constraint propagation
# For this graph, use natural order but try to find a good ordering
VERTEX_ORDER = list(range(N))
# Try to order by "centrality" - vertices that appear in more edges early
# Actually all have degree 4, so use index order

# Precompute for each vertex, its neighbors
def solve_with_k(k, deadline):
    """Try to find a valid k-coloring. Return list of colors or None."""
    color = [-1] * N
    
    # For each pair of colors (a, b), maintain DSU to detect cycles
    # pair_dsu[(a,b)] = (parent, rank, size) for the induced subgraph of colors a and b
    # We track: for each color pair, the DSU structure
    
    # More efficient: for each color pair, track edges and use DSU
    # pair_parent[(a,b)] = list of parent pointers for vertices colored a or b
    # pair_rank[(a,b)] = rank for union-find
    
    # Actually, let's use a simpler approach:
    # For each color pair (a,b), maintain:
    #   - set of vertices colored a or b
    #   - DSU over those vertices
    #   - count of edges in the induced subgraph
    # A bicolored cycle exists iff we add an edge within the same DSU component
    
    # State: for each unordered pair {a,b} with a<b, maintain DSU
    # We'll use dictionaries keyed by (min(a,b), max(a,b))
    
    import sys
    sys.setrecursionlimit(10000)
    
    # DSU structure for each color pair
    # pair_data[(a,b)] = {'parent': {v: p}, 'rank': {v: r}, 'edges': int, 'verts': set}
    # But this is expensive to copy. Let's use a different approach.
    
    # Alternative: maintain for each color pair a DSU with path compression
    # and track edge count. When we assign vertex v to color c:
    #   For each neighbor u already colored with color d != c:
    #     Check pair (c,d): if u and v are in same component, adding edge creates cycle -> invalid
    #     Else union them, increment edge count
    #   For each neighbor u already colored with color c:
    #     That's fine (same color, not bicolored)
    
    # Let me implement DSU per color pair carefully.
    
    # pair_parent: dict from (a,b) to dict mapping vertex -> parent
    # pair_rank: dict from (a,b) to dict mapping vertex -> rank
    # pair_edges: dict from (a,b) to int (number of edges in induced subgraph)
    
    # For efficiency, only create DSU entries when needed.
    
    def make_pair_key(a, b):
        if a > b:
            a, b = b, a
        return (a, b)
    
    pair_parent = {}  # (a,b) -> {v: parent}
    pair_rank = {}    # (a,b) -> {v: rank}
    pair_edges = {}   # (a,b) -> int
    
    def find(pair_key, v):
        parent = pair_parent[pair_key]
        while parent[v] != v:
            parent[v] = parent[parent[v]]  # path compression
            v = parent[v]
        return v
    
    def union(pair_key, u, v):
        parent = pair_parent[pair_key]
        rank = pair_rank[pair_key]
        ru = find(pair_key, u)
        rv = find(pair_key, v)
        if ru == rv:
            return False  # would create cycle
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        return True
    
    def assign(v, c):
        """Assign color c to vertex v. Return True if valid, False if creates bicolored cycle."""
        color[v] = c
        # For each neighbor u with a different color d
        for u in ADJ[v]:
            d = color[u]
            if d == -1:
                continue
            if d == c:
                continue  # same color, no bicolored constraint
            pk = make_pair_key(c, d)
            if pk not in pair_parent:
                # Initialize DSU for this pair
                pair_parent[pk] = {}
                pair_rank[pk] = {}
                pair_edges[pk] = 0
                # Add all existing vertices with colors c or d
                for w in range(N):
                    if color[w] == c or color[w] == d:
                        pair_parent[pk][w] = w
                        pair_rank[pk][w] = 0
            # Now add edge (v, u) to this pair's induced subgraph
            if not union(pk, v, u):
                # Would create a cycle -> invalid
                return False
            pair_edges[pk] += 1
        return True
    
    def unassign(v, c):
        """Undo assignment of color c to vertex v."""
        color[v] = -1
        for u in ADJ[v]:
            d = color[u]
            if d == -1:
                continue
            if d == c:
                continue
            pk = make_pair_key(c, d)
            if pk in pair_edges:
                pair_edges[pk] -= 1
            # Note: we don't undo the union in DSU for simplicity (this makes backtracking
            # not fully correct in terms of DSU state, but we re-initialize pairs fresh each time)
            # Actually this is a problem. Let me use a different approach.
    
    # The DSU undo is problematic. Let me use a simpler but correct approach:
    # Instead of maintaining DSU incrementally with undo, use a forward-checking approach
    # where we check validity by recomputing or by using a different data structure.
    
    # Alternative: for each color pair, just track the set of edges and check acyclicity
    # by counting: a forest on v vertices has at most v-1 edges.
    # So for pair (a,b): let V_ab = vertices colored a or b, E_ab = edges between them.
    # Valid iff E_ab <= |V_ab| - 1 for each connected component, or more simply:
    # E_ab <= |V_ab| - 1 is necessary but not sufficient (need per-component).
    # Actually, for a graph to be a forest, we need E = V - C where C is number of components.
    # Equivalently, no cycle. The simplest check: E_ab < |V_ab| is necessary but not sufficient.
    # We need the full cycle check.
    
    # Let me use a different strategy: maintain for each color pair a DSU, but when backtracking,
    # we save and restore the full DSU state. This is expensive but correct.
    
    # Actually, let's try a cleaner approach: use the fact that we can check if adding edge (v,u)
    # to the bicolored subgraph of colors (c,d) creates a cycle by checking if v and u are already
    # connected in that subgraph. We maintain DSU per pair. For backtracking, we save the state.
    
    # Let me restructure with explicit save/restore.
    
    # Given complexity, let me try a simpler approach first:
    # Greedy with backtracking, checking validity by full recomputation when needed.
    
    # Actually, let me just do a clean backtracking with full state save/restore for DSUs.
    
    # Reset state
    pair_parent = {}
    pair_rank = {}
    pair_edges = {}
    
    def get_or_init_pair(pk, vertices_with_colors):
        if pk not in pair_parent:
            pair_parent[pk] = {}
            pair_rank[pk] = {}
            pair_edges[pk] = 0
            a, b = pk
            for w in range(N):
                cw = color[w]
                if cw == a or cw == b:
                    pair_parent[pk][w] = w
                    pair_rank[pk][w] = 0
        return pair_parent[pk], pair_rank[pk]
    
    def find2(pk, v):
        parent = pair_parent[pk]
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    
    def union2(pk, u, v):
        parent = pair_parent[pk]
        rank = pair_rank[pk]
        ru = find2(pk, u)
        rv = find2(pk, v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        return True
    
    def is_valid_assignment(v, c):
        """Check if assigning color c to vertex v is valid (no bicolored cycle)."""
        for u in ADJ[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            pk = make_pair_key(c, d)
            # Check if u and v would be in same component in pair (c,d)
            # We need to check connectivity in the current induced subgraph
            # This requires the DSU to be up to date
            # For now, do a BFS/DFS check
            # Find all vertices with color c or d
            # Check if v and u are connected in the subgraph induced by colors c and d
            # (excluding the edge we're about to add, but including all existing edges)
            
            # BFS from u in the subgraph induced by colors c and d
            # v is not yet colored, so we check if there's a path from u to v
            # using only vertices colored c or d
            visited = set()
            stack = [u]
            visited.add(u)
            while stack:
                cur = stack.pop()
                if cur == v:
                    # v is reachable from u, so adding edge (v,u) creates a cycle
                    # Wait, v is not colored yet, so it's not in the subgraph.
                    # We need to check if there's a path from u to v through vertices
                    # colored c or d. Since v is uncolored, the path must end at v.
                    # But v isn't in the subgraph yet. So we check if any neighbor of v
                    # (colored c or d) is reachable from u.
                    pass
                for nb in ADJ[cur]:
                    if nb in visited:
                        continue
                    if nb == v:
                        # Found path to v, but v is uncolored. This means if we color v,
                        # it would connect to the component containing u.
                        # But we need to check if v is ALREADY connected to u via other paths.
                        # Since v is uncolored, it has no edges in the subgraph yet.
                        # So the only way adding (v,u) creates a cycle is if v is already
                        # in the same component as u through other edges.
                        # But v has no edges in the subgraph (it's uncolored), so v is isolated.
                        # Thus adding (v,u) just connects v to u's component. No cycle.
                        # UNLESS v is connected to u through another neighbor.
                        # So we need to check: is v already connected to u via another path?
                        # v's neighbors that are colored c or d: if any of them is in u's component,
                        # then adding (v,u) creates a cycle.
                        break
                    cw = color[nb]
                    if cw == -1:
                        continue
                    if cw != c and cw != d:
                        continue
                    visited.add(nb)
                    stack.append(nb)
            
            # Better approach: check if v has any neighbor (other than u) that is colored c or d
            # and is in the same component as u.
            for nb in ADJ[v]:
                if nb == u:
                    continue
                cn = color[nb]
                if cn == -1 or cn == c or cn == d:
                    if cn == c or cn == d:
                        # Check if nb is connected to u in the subgraph induced by c and d
                        # BFS from u to nb
                        if nb == u:
                            continue
                        visited2 = set()
                        stack2 = [u]
                        visited2.add(u)
                        found = False
                        while stack2:
                            cur2 = stack2.pop()
                            if cur2 == nb:
                                found = True
                                break
                            for nb2 in ADJ[cur2]:
                                if nb2 in visited2:
                                    continue
                                c2 = color[nb2]
                                if c2 == -1:
                                    continue
                                if c2 != c and c2 != d:
                                    continue
                                visited2.add(nb2)
                                stack2.append(nb2)
                        if found:
                            return False
        return True
    
    # This is getting complex. Let me use a simpler but potentially slower approach:
    # Full validation after each assignment.
    
    def full_validate():
        """Check if current partial coloring is valid (no bicolored cycles in any pair)."""
        from collections import defaultdict
        # For each pair of colors, check if induced subgraph is a forest
        color_pairs = defaultdict(list)  # (a,b) -> list of edges
        for u, v in EDGES:
            cu, cv = color[u], color[v]
            if cu == -1 or cv == -1:
                continue
            if cu == cv:
                continue
            pk = make_pair_key(cu, cv)
            color_pairs[pk].append((u, v))
        
        for pk, edges in color_pairs.items():
            a, b = pk
            # Get all vertices colored a or b
            verts = set()
            for u, v in edges:
                verts.add(u)
                verts.add(v)
            # Also add isolated vertices (colored a or b but no edges in this pair)
            for w in range(N):
                if color[w] == a or color[w] == b:
                    verts.add(w)
            # Check if the graph (verts, edges) is a forest
            # Use DSU
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def findf(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v
            
            for u, v in edges:
                ru = findf(u)
                rv = findf(v)
                if ru == rv:
                    return False  # cycle detected
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
        return True
    
    # Backtracking solver
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == N:
            return color[:]
        
        v = VERTEX_ORDER[idx]
        
        # Try each color
        for c in range(k):
            # Check if c is allowed (different from neighbors)
            valid = True
            for u in ADJ[v]:
                if color[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Assign and check
            color[v] = c
            if full_validate():
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            color[v] = -1
        
        return None
    
    result = backtrack(0)
    return result
    
    # If we get here, no solution found
    
# Fallback: try to find a good coloring using a heuristic
def heuristic_coloring(k):
    """Use a greedy heuristic to find a k-coloring, then repair."""
    color = [-1] * N
    # Order vertices by degree (all 4) then by index
    for v in range(N):
        # Find a color that doesn't conflict
        used = set()
        for u in ADJ[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(k):
            if c not in used:
                color[v] = c
                break
        else:
            # No valid color, try any
            color[v] = 0
    return color

def check_coloring(color):
    """Check if a coloring is valid."""
    from collections import defaultdict
    color_pairs = defaultdict(list)
    for u, v in EDGES:
        cu, cv = color[u], color[v]
        if cu == cv:
            continue
        pk = (min(cu, cv), max(cu, cv))
        color_pairs[pk].append((u, v))
    
    for pk, edges in color_pairs.items():
        a, b = pk
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        for w in range(N):
            if color[w] == a or color[w] == b:
                verts.add(w)
        parent = {v: v for v in verts}
        rank = {v: 0 for v in verts}
        
        def findf(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        
        for u, v in edges:
            ru = findf(u)
            rv = findf(v)
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Try k = 3, 4, 5, ...
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = solve_with_k(k, deadline)
        if result is not None:
            # Verify
            if check_coloring(result):
                return [int(c) for c in result]
    
    # Fallback: use heuristic with k=5
    color = heuristic_coloring(5)
    # Try to repair
    for _ in range(1000):
        if time.time() > deadline:
            break
        if check_coloring(color):
            break
        # Find a violation and fix it
        from collections import defaultdict
        color_pairs = defaultdict(list)
        for u, v in EDGES:
            cu, cv = color[u], color[v]
            if cu == cv:
                continue
            pk = (min(cu, cv), max(cu, cv))
            color_pairs[pk].append((u, v))
        
        fixed = False
        for pk, edges in color_pairs.items():
            a, b = pk
            verts = set()
            for u, v in edges:
                verts.add(u)
                verts.add(v)
            for w in range(N):
                if color[w] == a or color[w] == b:
                    verts.add(w)
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def findf(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v
            
            for u, v in edges:
                ru = findf(u)
                rv = findf(v)
                if ru == rv:
                    # Cycle found, try to recolor u
                    for c in range(5):
                        if c != color[u] and c != color[v]:
                            # Check if c conflicts with other neighbors
                            ok = True
                            for nb in ADJ[u]:
                                if color[nb] == c:
                                    ok = False
                                    break
                            if ok:
                                color[u] = c
                                fixed = True
                                break
                    if fixed:
                        break
        if not fixed:
            break
    
    # Final check and return
    if check_coloring(color):
        return [int(c) for c in color]
    
    # Last resort: return a valid 50-coloring
    return [int(i) for i in range(N)]
# EVOLVE-BLOCK-END
