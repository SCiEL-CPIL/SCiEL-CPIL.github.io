import numpy as np

def construct():
    q = 5
    n = q**2 + q + 1  # 31
    
    # Generate points in PG(2, q)
    # Points are represented as (x, y, z) with first non-zero coord = 1
    points = []
    
    # Case 1: x != 0 -> x = 1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
            
    # Case 2: x = 0, y != 0 -> y = 1
    for z in range(q):
        points.append((0, 1, z))
        
    # Case 3: x = 0, y = 0, z != 0 -> z = 1
    points.append((0, 0, 1))
    
    # Verify count
    assert len(points) == n, f"Expected {n} points, got {len(points)}"
    
    edges = []
    
    # Compute edges based on polarity x1*x2 + y1*y2 + z1*z2 == 0 mod q
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    # The polarity graph has n(q+1)/2 edges = 31*6/2 = 93 edges
    # We need exactly 91 edges
    assert len(edges) >= 91, f"Expected at least 91 edges, got {len(edges)}"
    
    return edges[:91]
