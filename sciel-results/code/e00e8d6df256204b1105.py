import numpy as np
import time
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s kill
    
    q = 5
    
    # Enumerate all points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first nonzero coordinate = 1
                if x != 0:
                    inv_x = pow(x, 3, 5)  # x^4 = 1 mod 5
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    n = 31
    
    # Try both polarities and pick the one that gives 90 edges and is C4-free
    # Polarity 1: x1*x2 + y1*y2 + z1*z2 = 0
    # Polarity 2: x1*x2 + y1*y2 - z1*z2 = 0 (i.e., x1*x2 + y1*y2 + 4*z1*z2 = 0 mod 5)
    
    def build_graph(sign_z):
        """Build adjacency matrix. sign_z = 1 for +z1*z2, sign_z = 4 for -z1*z2 (mod 5)"""
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            x1, y1, z1 = points[i]
            for j in range(i + 1, n):
                x2, y2, z2 = points[j]
                val = (x1 * x2 + y1 * y2 + sign_z * z1 * z2) % 5
                if val == 0:
                    A[i, j] = 1
                    A[j, i] = 1
        return A
    
    def count_c4_fast(A):
        """Count C4s using: (1/4) * sum_{u<v} C(cn(u,v), 2)"""
        # Common neighbors count for each pair
        # cn(u,v) = sum_w A[u,w]*A[v,w]
        # Use matrix multiplication: (A @ A.T)[u,v] = number of common neighbors
        A2 = A @ A.T
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                cn = A2[u, v]
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 4
    
    # Try both polarities
    best_core = None
    best_c4 = float('inf')
    
    for sign_z in [1, 4]:
        A = build_graph(sign_z)
        n_edges = A.sum() // 2
        if n_edges != 90:
            continue
        c4_count = count_c4_fast(A)
        if c4_count < best_c4:
            best_c4 = c4_count
            best_core = A.copy()
    
    if best_core is None:
        # Fallback: use sign_z=1
        best_core = build_graph(1)
    
    A = best_core
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find the non-edge that adds the minimum number of C4s
    # Adding edge (u,v) creates C4s = number of paths of length 3 from u to v
    # = number of pairs (a,b) where a in N(u), b in N(v), and a~b
    
    # Precompute: for each vertex, its neighbors as a numpy array for fast intersection
    # A3[u,v] = number of length-3 paths from u to v
    # A3 = A @ A @ A
    A2 = A @ A
    A3 = A2 @ A
    
    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u + 1, n):
            if A[u, v]:
                continue
            # Number of length-3 paths from u to v
            add = A3[u, v]
            if add < best_add:
                best_add = add
                best_edge = (u, v)
    
    if best_edge is None:
        # Fallback: find any non-edge
        for u in range(n):
            for v in range(u + 1, n):
                if not A[u, v]:
                    best_edge = (u, v)
                    break
            if best_edge:
                break
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges.append([i, j])
    
    edges.append([best_edge[0], best_edge[1]])
    
    # Validate
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    edge_set = set()
    for e in edges:
        key = tuple(sorted(e))
        assert key not in edge_set, f"Duplicate edge {key}"
        edge_set.add(key)
    
    return edges
