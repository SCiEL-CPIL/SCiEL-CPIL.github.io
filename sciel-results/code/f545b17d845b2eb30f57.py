import itertools
import time

def construct():
    q = 5
    # Generate points of PG(2, q)
    # Represent points as tuples (x, y, z) with gcd(x,y,z)=1 and x>=0, etc.
    # Standard representative: x in 1..q-1, y,z in 0..q-1, plus points with x=0.
    # Easier: iterate all non-zero vectors (x,y,z) in F_q^3, normalize.
    
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: find first non-zero coordinate, make it 1
                # If x != 0, (1, y/x, z/x)
                # If x == 0, y != 0, (0, 1, z/y)
                # If x == 0, y == 0, z != 0, (0, 0, 1)
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    rep = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    rep = (0, 1, (z * inv_y) % q)
                else:
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    n = len(points)
    # n should be 31
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            # Check orthogonality: x1*x2 + y1*y2 + z1*z2 == 0 mod q
            val = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if val == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
                
    # Verify edge count
    # Should be 90
    
    # For each non-edge, count length-3 paths
    # Path u-w1-w2-v means w1 in N(u), w2 in N(v), w1 ~ w2
    # We want to minimize this count.
    
    min_paths = float('inf')
    best_edge = None
    
    # Precompute neighbors
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                neighbors[i].append(j)
                
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                # Count paths
                count = 0
                for w1 in neighbors[i]:
                    for w2 in neighbors[j]:
                        if w1 != w2 and adj[w1][w2]:
                            count += 1
                if count < min_paths:
                    min_paths = count
                    best_edge = [i, j]
                    
    if best_edge:
        edges.append(best_edge)
        
    return edges
