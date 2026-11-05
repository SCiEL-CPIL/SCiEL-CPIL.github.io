"""Construct C4-free graph with no independent 11-set; maximize n."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _verify_c4_free(n, edges):
    """Return True if graph has no C4 (no two vertices with >=2 common neighbors)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set_of_size_k(n, adj, k):
    """Backtracking search for independent set of size k. Returns True if found."""
    # Use greedy coloring bound for pruning (Bron–Kerbosch style)
    # Simple backtracking with ordering
    best = [0]  # current max independent set size found so far (for early termination)
    
    def backtrack(remaining, current_size, current_set):
        if current_size + len(remaining) <= current_size:  # can't beat
            return False
        if current_size == k:
            return True
        # If even taking all remaining can't reach k
        if current_size + len(remaining) < k:
            return False
        # Try greedy: pick vertex with min degree in remaining
        # For speed, just iterate
        for idx in range(len(remaining)):
            v = remaining[idx]
            # Check if v is independent of current_set
            if all(v not in adj[u] for u in current_set):
                new_remaining = [x for x in remaining[idx+1:] if x not in adj[v]]
                if backtrack(new_remaining, current_size + 1, current_set + [v]):
                    return True
        return False
    
    return backtrack(list(range(n)), 0, [])

def _polarity_graph(q):
    """
    Construct polarity graph of PG(2, q) for prime q.
    Vertices: q^2 + q + 1 points of PG(2,q).
    Edge between u,v iff u · v = 0 mod q (orthogonal under standard bilinear form).
    This graph is C4-free.
    """
    # Points of PG(2, q): represented as (x, y, z) not all zero, modulo scaling
    # Use canonical form: for affine points z=1, (x, y, 1); for line at infinity, (1, m, 0) and (0,1,0)
    points = []
    # Affine points: (x, y, 1) for x,y in 0..q-1
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity: (1, m, 0) for m in 0..q-1, and (0, 1, 0)
    for m in range(q):
        points.append((1, m, 0))
    points.append((0, 1, 0))
    
    n_total = len(points)  # q^2 + q + 1
    
    # Bilinear form: (x1,y1,z1)·(x2,y2,z2) = x1*x2 + y1*y2 + z1*z2 mod q
    # Actually for polarity graph of PG(2,q) with standard polarity,
    # the orthogonal condition is: x1*x2 + y1*y2 + z1*z2 ≡ 0 (mod q)
    # But we need to be careful: this is the standard dot product.
    # For q=7, this works.
    
    edges = []
    adj = [set() for _ in range(n_total)]
    
    for i in range(n_total):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n_total):
            x2, y2, z2 = points[j]
            dot = (x1 * x2 + y1 * y2 + z1 * z2) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return n_total, edges, adj

def construct():
    start_time = time.time()
    deadline = start_time + 120  # must finish within 120s per task, use 150s wall clock guard
    
    best_n = 0
    best_edges = []
    
    # Try polarity graphs for primes q where q^2 + q + 1 >= 30
    # q=5: 31 vertices, q=7: 57 vertices
    for q in [5, 7, 11]:
        if time.time() > deadline - 5:
            break
        
        n_total, edges_full, adj_full = _polarity_graph(q)
        
        # Check C4-free (should be guaranteed, but verify)
        if not _verify_c4_free(n_total, edges_full):
            continue
        
        # We want to find largest induced subgraph with alpha <= 10
        # For q=7, n_total=57, alpha of full graph is about 8 (known result)
        # So any induced subgraph also has alpha <= 8 <= 10
        # We can take up to min(n_total, 43) vertices
        
        n_target = min(n_total, 43)
        
        if n_target <= best_n:
            continue
        
        # Take first n_target vertices
        selected = list(range(n_target))
        
        # Build induced subgraph
        sel_to_idx = {v: i for i, v in enumerate(selected)}
        edges = []
        for u, v in edges_full:
            if u in sel_to_idx and v in sel_to_idx:
                edges.append([sel_to_idx[u], sel_to_idx[v]])
        
        # Verify
        if not _verify_c4_free(n_target, edges):
            continue
        
        # Check independent set of size 11
        adj_sub = [set() for _ in range(n_target)]
        for u, v in edges:
            adj_sub[u].add(v)
            adj_sub[v].add(u)
        
        if _has_independent_set_of_size_k(n_target, adj_sub, 11):
            continue
        
        best_n = n_target
        best_edges = edges
        break  # got 43, maximum possible
    
    # If we didn't get 43, try smaller constructions
    if best_n < 30:
        # Fallback: 10 disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best_n = 30
        best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
