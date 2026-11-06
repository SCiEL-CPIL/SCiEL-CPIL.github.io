import itertools
import time

def construct():
    """
    Constructs a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    Uses the polarity graph of PG(2, 5) which has 90 edges and 0 4-cycles.
    Adds the single edge that creates the fewest new 4-cycles.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate vertices of PG(2, 5)
    # Points are (x, y, z) in F5^3, not all zero, up to scalar.
    # We use unique representatives.
    vertices = []
    
    # z = 1
    for x in range(5):
        for y in range(5):
            vertices.append((x, y, 1))
            
    # z = 0, y = 1
    for x in range(5):
        vertices.append((x, 1, 0))
        
    # z = 0, y = 0, x = 1
    vertices.append((1, 0, 0))
    
    n = len(vertices)
    assert n == 31, f"Expected 31 vertices, got {n}"
    
    # 2. Build adjacency matrix for the polarity graph
    # Polarity: x1*x2 + y1*y2 + z1*z2 == 0 (mod 5)
    A = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            p = vertices[i]
            q = vertices[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                A[i][j] = 1
                A[j][i] = 1
                
    # Count edges to verify
    edge_count = sum(sum(row) for row in A) // 2
    assert edge_count == 90, f"Expected 90 edges, got {edge_count}"
    
    # 3. Compute A^3 to find number of length-3 paths
    # A^2 = A * A
    A2 = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k]:
                aik = A[i][k]
                for j in range(n):
                    A2[i][j] += aik * A[k][j]
                    
    # A^3 = A^2 * A
    A3 = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A2[i][k]:
                a2ik = A2[i][k]
                for j in range(n):
                    A3[i][j] += a2ik * A[k][j]
                    
    # 4. Find non-edge (u, v) with minimum A3[u][v]
    min_paths = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i+1, n):
            if A[i][j] == 0:
                paths = A3[i][j]
                if paths < min_paths:
                    min_paths = paths
                    best_edge = (i, j)
                    
    if best_edge is None:
        # Fallback: should not happen
        best_edge = (0, 1)
        
    # 5. Construct final list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j] == 1:
                edges.append([i, j])
                
    # Add the new edge
    edges.append([best_edge[0], best_edge[1]])
    
    # Sort for consistency (optional, but good for determinism)
    edges.sort()
    
    assert len(edges) == 91
    assert len(set(map(tuple, edges))) == 91
    
    return edges

if __name__ == "__main__":
    construct()
