"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    Construct the polarity graph of PG(2,5) with 90 edges.
    
    For q=5 (odd), the orthogonal polarity graph of PG(2,q) has:
    - n = q^2 + q + 1 = 31 vertices
    - 90 edges (C4-free)
    """
    q = 5
    # Enumerate all 1D subspaces of F_q^3
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
    
    # Use the symmetric matrix M = [[1,0,0],[0,1,0],[0,0,-1]] mod 5
    # This gives a polarity where x^T M y = 0 defines perpendicularity
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Build the polarity graph (C4-free, 90 edges)
    edges_90, adj = _build_polarity_graph()
    
    # Count current C4s (should be 0)
    c4_base = _count_c4(adj)
    
    # Find the best edge to add (the one that creates the fewest C4s)
    best_edge = None
    best_c4 = float('inf')
    
    n = 31
    # Try all possible edges not in the graph
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Adding edge (i,j) creates C4s for each path of length 3 between i and j
                # A C4 containing the new edge (i,j) has the form i-a-b-j-i where a is neighbor of i,
                # b is neighbor of j, a != b, and (a,b) is an edge.
                
                ni = np.where(adj[i] == 1)[0]
                nj = np.where(adj[j] == 1)[0]
                
                c4s_created = 0
                for a in ni:
                    for b in nj:
                        if a != b and adj[a, b] == 1:
                            c4s_created += 1
                
                total_c4 = c4_base + c4s_created
                
                if total_c4 < best_c4:
                    best_c4 = total_c4
                    best_edge = [i, j]
                
                # Check time
                if time.time() > deadline - 5.0:
                    break
        if time.time() > deadline - 5.0:
            break
    
    # If we found a good edge, add it
    if best_edge is not None and time.time() < deadline - 2.0:
        edges_90.append(best_edge)
        return edges_90
    else:
        # Fallback: add the first available edge
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    edges_90.append([i, j])
                    return edges_90
    
    return edges_90

# EVOLVE-BLOCK-END
