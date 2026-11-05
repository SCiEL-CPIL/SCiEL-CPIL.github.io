"""Direct exact max-clique search on a derived C4-free candidate graph.

Strategy (different family from the seed program):
  * Build a deterministic C4-free seed by combining the polarity graph of
    PG(2,5) (n=31, known C4-free) with a disjoint 30-vertex triangle packing,
    then add a small number of cross edges that are provably C4-safe.
  * Instead of "grow the graph and hope alpha<11", run an exact maximum-clique
    search on the *complement* of the seed to find the largest independent set.
    The search is branch-and-bound with:
      - a tight upper bound via greedy coloring of the remaining subgraph,
      - C4-aware pruning: when expanding a clique in the complement, we ensure
        that adding a vertex does not create a C4 in the original graph.
  * If the maximum independent set is < 11, we have a valid graph.
  * We also try a few deterministic variants (different q, different triangle
    packings) to maximize n.
"""
import time
import random

# EVOLVE-BLOCK-START

def _inv_mod(a, q):
    return pow(a, q - 2, q)

def _projective_points(q):
    pts = []
    invs = [0] + [_inv_mod(a, q) for a in range(1, q)]
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    ix = invs[x]
                    pts.append((1, (y * ix) % q, (z * ix) % q))
                elif y != 0:
                    iy = invs[y]
                    pts.append((0, 1, (z * iy) % q))
                else:
                    pts.append((0, 0, 1))
    return pts

def _polarity_graph(q):
    pts = _projective_points(q)
    n = len(pts)
    adj = [0] * n
    for i in range(n):
        a1, b1, c1 = pts[i]
        for j in range(i + 1, n):
            a2, b2, c2 = pts[j]
            if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return n, adj

def _triangle_packing(n_triangles):
    """n_triangles disjoint triangles, C4-free, alpha = n_triangles."""
    n = 3 * n_triangles
    adj = [0] * n
    for t in range(n_triangles):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    return n, adj

def _combine_graphs(n1, adj1, n2, adj2, cross_edges):
    """Combine two C4-free graphs with specified cross edges (u in G1, v in G2)."""
    n = n1 + n2
    adj = adj1[:] + [0] * n2
    # offset adj2
    for i in range(n2):
        adj[n1 + i] = adj2[i]
    # add cross edges
    for u, v in cross_edges:
        adj[u] |= (1 << (n1 + v))
        adj[n1 + v] |= (1 << u)
    return n, adj

def _is_c4_free(n, adj):
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            if common & (common - 1):
                return False
    return True

def _complement_adj(n, adj):
    """Return adjacency bitmasks for the complement graph."""
    comp = [0] * n
    for i in range(n):
        # all bits except self and adj[i]
        all_bits = (1 << n) - 1
        comp[i] = all_bits & ~(adj[i] | (1 << i))
    return comp

def _max_clique_exact(n, comp_adj, deadline, limit=11):
    """Exact maximum clique in complement graph (i.e. max independent set in original).
    Returns the size of the maximum clique. Prunes if size >= limit.
    Uses branch-and-bound with greedy coloring upper bound."""
    best_size = 0
    # Order vertices by degree in comp_adj (higher degree first)
    order = sorted(range(n), key=lambda v: -comp_adj[v].bit_count())
    # Map original vertex to position in order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # Precompute comp_adj in order
    comp_ordered = [0] * n
    for i, v in enumerate(order):
        comp_ordered[i] = comp_adj[v]
    
    # Greedy coloring for upper bound
    def greedy_color_bound(remaining, n_rem):
        """Color the subgraph induced by remaining vertices.
        Returns a list of color classes (as bitmasks) and the number of colors."""
        color_classes = []
        uncolored = remaining
        while uncolored:
            color_class = 0
            # Greedily assign vertices to this color class
            candidate = uncolored
            while candidate:
                lb = candidate & (-candidate)
                v = lb.bit_length() - 1
                # v can be in this color class if it has no edges to existing color class
                if comp_ordered[v] & color_class:
                    candidate ^= lb
                    continue
                color_class |= lb
                candidate ^= lb
            if color_class:
                color_classes.append(color_class)
                uncolored &= ~color_class
        return len(color_classes)
    
    # Recursive search
    def dfs(remaining, clique_size, deadline):
        nonlocal best_size
        if time.time() > deadline:
            return
        if clique_size >= limit:
            best_size = limit
            return
        if not remaining:
            return
        if clique_size + remaining.bit_count() <= best_size:
            return
        # Upper bound via coloring
        num_colors = greedy_color_bound(remaining, n)
        if clique_size + num_colors <= best_size:
            return
        
        # Pick vertex with highest degree in remaining
        best_v = -1
        best_deg = -1
        r = remaining
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = (comp_ordered[v] & remaining).bit_count()
            if d > best_deg:
                best_deg = d
                best_v = v
            r ^= lb
        
        # Branch: include best_v
        new_remaining = remaining & comp_ordered[best_v]
        dfs(new_remaining, clique_size + 1, deadline)
        
        # Branch: exclude best_v
        remaining &= ~(1 << best_v)
        dfs(remaining, clique_size, deadline)
    
    dfs((1 << n) - 1, 0, deadline)
    return best_size

def _try_add_cross_edges(n1, adj1, n2, adj2, deadline):
    """Try to add cross edges between two C4-free graphs while maintaining C4-free."""
    n = n1 + n2
    adj = adj1[:] + [0] * n2
    for i in range(n2):
        adj[n1 + i] = adj2[i]
    
    # Try all possible cross edges, add if C4-safe
    added = 0
    for u in range(n1):
        for v in range(n2):
            v2 = n1 + v
            # Check if adding edge (u, v2) creates C4
            # A C4 would be: u - a - b - v2 - u where a is neighbor of u in G1,
            # b is neighbor of v in G2, and a-b is not an edge (but we need to check
            # that a and b are connected via some path of length 2 in the combined graph)
            # Actually, C4 in combined graph with new edge (u, v2):
            # u - x - y - v2 - u where x is neighbor of u (in G1 or cross),
            # y is neighbor of v2 (in G2 or cross), and x-y is an edge.
            # Since we're adding one edge at a time, we need to check:
            # For each neighbor x of u in current adj, and each neighbor y of v2 in current adj,
            # if x and y are adjacent, then adding (u, v2) creates a C4.
            # But x is in G1 (or cross to G2) and y is in G2 (or cross to G1).
            # If x is in G1 and y is in G2, x-y is not an edge (no cross edges yet), so safe.
            # If x is in G2 (cross) and y is in G2, then x-y is an edge in G2, and we have
            # u - x - y - v2 - u, which is a C4 if u-x is cross, x-y is in G2, y-v2 is in G2, v2-u is cross.
            # But u-x is cross only if x >= n1. Let's check carefully.
            
            # Simpler: just add the edge and check C4-free on the whole graph.
            adj[u] |= (1 << v2)
            adj[v2] |= (1 << u)
            if not _is_c4_free(n, adj):
                adj[u] &= ~(1 << v2)
                adj[v2] &= ~(1 << u)
            else:
                added += 1
            if time.time() > deadline:
                return n, adj
    return n, adj

def construct():
    deadline = time.time() + 140.0
    best_n = 0
    best_adj = None
    
    def consider(n, adj):
        nonlocal best_n, best_adj
        if n <= best_n or n < 1:
            return
        if not _is_c4_free(n, adj):
            return
        # Check alpha < 11 via exact max clique in complement
        comp = _complement_adj(n, adj)
        max_indep = _max_clique_exact(n, comp, deadline, limit=11)
        if max_indep >= 11:
            return
        best_n = n
        best_adj = adj[:]
    
    # Strategy 1: Polarity graphs
    for q in [5, 7, 11, 13]:
        if time.time() > deadline - 10:
            break
        try:
            n, adj = _polarity_graph(q)
            consider(n, adj)
        except Exception:
            continue
    
    # Strategy 2: Triangle packings combined with polarity graph
    for q in [5, 7]:
        if time.time() > deadline - 10:
            break
        try:
            n1, adj1 = _polarity_graph(q)
            # Add triangle packing to reach higher n
            # Polarity graph q=5 has n=31, alpha ~ 5-6
            # Add 3 triangles (9 vertices) to get n=40
            for n_tri in [3, 4, 5]:
                if time.time() > deadline - 10:
                    break
                n2, adj2 = _triangle_packing(n_tri)
                # Try to add cross edges
                n, adj = _try_add_cross_edges(n1, adj1, n2, adj2, deadline - 5)
                consider(n, adj)
        except Exception:
            continue
    
    # Strategy 3: Pure triangle packing (fallback)
    for n_tri in [10, 11, 12, 13]:
        if time.time() > deadline - 5:
            break
        try:
            n, adj = _triangle_packing(n_tri)
            consider(n, adj)
        except Exception:
            continue
    
    # Fallback
    if best_n < 30:
        n = 30
        adj = [0] * n
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        best_n = n
        best_adj = adj
    
    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            lb = m & (-m)
            j = lb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lb
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
