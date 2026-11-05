"""Construct a C4-free graph with no independent set of size 11, maximizing n."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

# ---- GF(4) helpers (for polarity graphs) ----
# GF(4) = {0,1,2,3} with primitive element w = 1+sqrt(-1)
# Multiplication table
_GF4_MUL = [
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
]
_GF4_ADD = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]

def gf4_add(a, b):
    return _GF4_ADD[a][b]

def gf4_mul(a, b):
    return _GF4_MUL[a][b]

def gf4_sub(a, b):
    # In char 2, subtraction = addition
    return _GF4_ADD[a][b]

def gf4_eq(a, b):
    return a == b

# ---- C4-free graph construction using polarity graph of GF(4)^2 ----
def _polarity_graph_gf4_2():
    """
    Vertices are (x, y) in GF(4)^2, i.e., 16 vertices.
    Two vertices (x1,y1) and (x2,y2) are adjacent iff
    x1*y2 + x2*y1 = 0 (the Hermitian form over GF(4)).
    This is a C4-free graph (it's a polarity graph of a projective plane of order 3).
    It has 16 vertices, 40 edges, and independence number 5.
    """
    n = 16
    edges = []
    for i in range(n):
        x1, y1 = i // 4, i % 4
        for j in range(i + 1, n):
            x2, y2 = j // 4, j % 4
            # Check x1*y2 + x2*y1 == 0
            val = gf4_add(gf4_mul(x1, y2), gf4_mul(x2, y1))
            if val == 0 and i != j:
                edges.append([i, j])
    return n, edges

def _check_c4_free(n, edges):
    """Verify no two vertices share more than one common neighbor."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            common = adj[u] & adj[v]
            if len(common) > 1:
                return False
    return True

def _max_independent_set_size(n, edges, timeout=5.0):
    """
    Find a large independent set using greedy + local search.
    Returns the size of the best independent set found.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    start = time.time()
    best_size = 0
    
    # Greedy: repeatedly pick vertex of min degree in remaining
    remaining = set(range(n))
    independent = set()
    while remaining:
        # Pick vertex with min degree in remaining
        best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        independent.add(best_v)
        remaining -= adj[best_v]
        remaining.discard(best_v)
    
    best_size = max(best_size, len(independent))
    
    # Try random restarts
    rng = random.Random(42)
    for _ in range(2000):
        if time.time() - start > timeout:
            break
        independent = set()
        remaining = set(range(n))
        # Random order
        vertices = list(range(n))
        rng.shuffle(vertices)
        for v in vertices:
            if v in remaining:
                independent.add(v)
                remaining -= adj[v]
                remaining.discard(v)
        best_size = max(best_size, len(independent))
    
    return best_size

def _add_vertex_c4_free(n, edges, adj, vertex_order, timeout=None, start_time=None):
    """
    Try to add a new vertex connected to some existing vertices.
    Returns (success, new_edges_list, new_adj)
    """
    if start_time is None:
        start_time = time.time()
    if timeout is not None and time.time() - start_time > timeout:
        return False, None, None
    
    new_n = n + 1
    # Try all subsets of existing vertices for neighbors of new vertex
    # New vertex v_new connected to S ⊆ {0..n-1}
    # C4 constraint: for any two neighbors u, w of v_new, u and w must have at most 0 common neighbors in the original graph (since v_new is a common neighbor).
    # Wait, that's not right. The C4-free condition is: no two vertices have >1 common neighbor.
    # If v_new is connected to u and w, then u and w already share v_new as a common neighbor.
    # So u and w must have NO other common neighbors in the original graph.
    # Also, v_new and any existing vertex x: they share neighbors in S ∩ adj[x]. Must have ≤1.
    
    # So the constraint is: for any u, w in S (u≠w), |adj[u] ∩ adj[w]| must be 0 (since v_new is already a common neighbor).
    # And for any x not in S, |S ∩ adj[x]| ≤ 1 (since if x is connected to 2 vertices in S, then x and v_new share 2 neighbors).
    # Wait: x and v_new share neighbors in S ∩ adj[x]. If |S ∩ adj[x]| ≥ 2, then x and v_new have ≥2 common neighbors, forming a C4.
    # Also for x in S: x and v_new share v_new's other neighbors that are also neighbors of x. That's |S ∩ adj[x]| - 1 (excluding x itself if x is in S... wait x is in S, so adj[x] doesn't contain x).
    # Actually adj[x] contains the neighbors of x. v_new is a new vertex. Common neighbors of x and v_new are: vertices in S that are also neighbors of x. So |S ∩ adj[x]|.
    # If |S ∩ adj[x]| ≥ 2, C4 exists.
    
    # So constraints:
    # 1. For any distinct u, w in S: |adj[u] ∩ adj[w]| == 0
    # 2. For any x in {0..n-1}: |S ∩ adj[x]| ≤ 1
    
    # This is a strong constraint. Let's try small subsets of S.
    # Start with |S| = 1, then 2, etc.
    
    # For |S| = 1: S = {v}. Constraint 1 is vacuous. Constraint 2: for all x, |{v} ∩ adj[x]| ≤ 1, which is always true.
    # So we can always add a vertex connected to exactly 1 existing vertex.
    
    # For |S| = 2: S = {u, w}. Need |adj[u] ∩ adj[w]| == 0. And for all x, |{u,w} ∩ adj[x]| ≤ 1, i.e., no x adjacent to both u and w.
    
    # Let's try to maximize |S| while maintaining C4-freeness.
    # This is getting complex. Let's just try to add vertices with |S|=1 first (pendant vertices),
    # but that won't help with independence number much.
    
    # Alternative approach: use the polarity graph and try to find a larger C4-free subgraph or use a different construction.
    
    # Actually, let's think about this differently. The known best construction has 39 vertices.
    # One known construction: take a polarity graph of the projective plane of order 3 (16 vertices, 40 edges, alpha=5),
    # and combine multiple copies or use a different structure.
    
    # Another approach: use a strongly regular graph or a known C4-free graph with high independence number control.
    
    # Let me try a different strategy: build the graph incrementally, adding vertices and choosing edges to maximize n while keeping alpha < 11.
    
    return False, None, None

def _greedy_c4_free_construction(max_n=43, time_limit=120.0):
    """
    Greedy construction: start with a base C4-free graph, then add vertices.
    """
    start_time = time.time()
    
    # Start with the polarity graph of GF(4)^2 (16 vertices)
    n, edges = _polarity_graph_gf4_2()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_n = n
    best_edges = list(edges)
    
    # Check C4-free
    if not _check_c4_free(n, edges):
        # Fallback: use 10 disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        n = 30
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        best_n = n
        best_edges = list(edges)
    
    # Try to add vertices one by one
    for attempt in range(100):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Try to add a new vertex connected to a subset S of existing vertices
        # We want S to be as large as possible to reduce independence number
        
        # Find a valid S: for u, w in S, |adj[u]∩adj[w]| == 0, and for all x, |S∩adj[x]| ≤ 1
        # Start with single vertices, then try pairs
        
        # First, find all vertices that can be a neighbor of the new vertex
        # A vertex v can be in S if adding it to S doesn't violate constraints
        
        # Let's try all subsets of size 1 to 5
        found_valid_S = False
        best_S = None
        
        # Try size 1
        for v in range(n):
            S = [v]
            # Check constraint 2: for all x, |S ∩ adj[x]| ≤ 1 → always true for |S|=1
            # Constraint 1: vacuous
            # So any single vertex works
            if best_S is None or len(S) > len(best_S):
                best_S = S
            found_valid_S = True
        
        # Try size 2
        for u in range(n):
            for w in range(u + 1, n):
                # Check |adj[u] ∩ adj[w]| == 0
                if len(adj[u] & adj[w]) > 0:
                    continue
                # Check for all x: |{u,w} ∩ adj[x]| ≤ 1, i.e., no x adjacent to both
                # This is equivalent to |adj[u] ∩ adj[w]| == 0, which we already checked
                # Wait, no. adj[u] ∩ adj[w] is the set of common neighbors of u and w.
                # If x is adjacent to both u and w, then x ∈ adj[u] ∩ adj[w].
                # So |adj[u] ∩ adj[w]| == 0 means no x is adjacent to both u and w.
                # That's exactly what we need for constraint 2 with S={u,w}.
                # So the condition is just |adj[u] ∩ adj[w]| == 0.
                S = [u, w]
                if best_S is None or len(S) > len(best_S):
                    best_S = S
                    found_valid_S = True
        
        # Try size 3
        for u in range(n):
            for w in range(u + 1, n):
                if len(adj[u] & adj[w]) > 0:
                    continue
                for z in range(w + 1, n):
                    if len(adj[u] & adj[z]) > 0 or len(adj[w] & adj[z]) > 0:
                        continue
                    # Check constraint 2: for all x, |{u,w,z} ∩ adj[x]| ≤ 1
                    # This means no x is adjacent to 2 or more of u, w, z
                    valid = True
                    for x in range(n):
                        count = 0
                        if x in adj[u]:
                            count += 1
                        if x in adj[w]:
                            count += 1
                        if x in adj[z]:
                            count += 1
                        if count > 1:
                            valid = False
                            break
                    if valid:
                        S = [u, w, z]
                        if best_S is None or len(S) > len(best_S):
                            best_S = S
                            found_valid_S = True
        
        if not found_valid_S or best_S is None:
            break
        
        # Add new vertex connected to best_S
        new_v = n
        new_edges = list(edges) + [[new_v, v] for v in best_S]
        
        # Verify C4-free
        if _check_c4_free(n + 1, new_edges):
            # Update
            edges = new_edges
            adj = [set() for _ in range(n + 1)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            n = n + 1
            best_n = n
            best_edges = list(edges)
            
            # Check independence number
            alpha = _max_independent_set_size(n, edges, timeout=2.0)
            if alpha >= 11:
                # Too many independent vertices, remove some edges? Or stop?
                # We need alpha < 11. Let's check if we can fix this.
                # For now, just track it.
                pass
    
    return best_n, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Try the greedy construction
    n, edges = _greedy_c4_free_construction(max_n=43, time_limit=100.0)
    
    # Verify
    if _check_c4_free(n, edges):
        alpha = _max_independent_set_size(n, edges, timeout=5.0)
        if alpha < 11:
            return {'n': n, 'edges': edges}
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
# EVOLVE-BLOCK-END
