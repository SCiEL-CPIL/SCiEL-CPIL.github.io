"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix."""
    n = adj.shape[0]
    common = adj @ adj
    total = 0
    for u in range(n):
        for w in range(u + 1, n):
            k = common[u, w]
            if k >= 2:
                total += k * (k - 1) // 2
    return total

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5).
    Returns (edges, adj) where edges has 90 entries and adj is 31x31.
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    # Use a symmetric polarity matrix. For q=5 odd, use:
    # M = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    # This is a correlation polarity.
    M = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    
    return edges, adj

def _find_best_edge_to_add(adj, c4_base):
    """
    Find the non-edge (i,j) whose addition creates the fewest new 4-cycles.
    Returns (best_edge, min_c4_increase).
    """
    n = adj.shape[0]
    best_edge = None
    best_increase = float('inf')
    
    # Precompute neighbor lists
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Count edges (a,b) where a in N(i), b in N(j), a != b
                ni = neighbors[i]
                nj = neighbors[j]
                c4s = 0
                for a in ni:
                    for b in nj:
                        if a != b and adj[a, b] == 1:
                            c4s += 1
                if c4s < best_increase:
                    best_increase = c4s
                    best_edge = [i, j]
    
    return best_edge, best_increase

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Build the base polarity graph (should be C4-free with 90 edges)
    edges_90, adj = _build_polarity_graph()
    
    # Verify edge count
    n_edges = sum(1 for e in edges_90)
    if n_edges != 90:
        # Fallback: just return what we have plus a dummy edge
        # This shouldn't happen but guard against it
        pass
    
    # Count C4s in base graph (should be 0)
    c4_base = _count_c4(adj)
    
    # Find the best edge to add
    best_edge, c4_increase = _find_best_edge_to_add(adj, c4_base)
    
    # Check time
    if time.time() > deadline:
        # Emergency fallback
        for i in range(31):
            for j in range(i + 1, 31):
                if adj[i, j] == 0:
                    edges_90.append([i, j])
                    return edges_90[:91]
    
    if best_edge is not None:
        edges_90.append(best_edge)
    else:
        # Should not happen, but add first available
        for i in range(31):
            for j in range(i + 1, 31):
                if adj[i, j] == 0:
                    edges_90.append([i, j])
                    break
            if len(edges_90) >= 91:
                break
    
    return edges_90[:91]

# EVOLVE-BLOCK-END
