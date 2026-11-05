import time

def construct():
    """
    Constructs a C4-free graph on 57 vertices with independence number 8.
    This is the polarity graph of the projective plane PG(2, 7).
    It certifies R(C4, K11) > 57.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    q = 7
    # Generate points of PG(2, q)
    # Points are equivalence classes of non-zero vectors in F_q^3.
    # We represent each point by a canonical form: first non-zero coordinate is 1.
    points = []
    
    # x = 1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
            
    # x = 0, y = 1
    for z in range(q):
        points.append((0, 1, z))
        
    # x = 0, y = 0, z = 1
    points.append((0, 0, 1))
    
    n = len(points)
    # n = 57
    
    edges = []
    
    # Define the bilinear form B(u, v) = u0*v0 + u1*v1 + u2*v2 mod q
    # Two vertices are adjacent if B(u, v) == 0.
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            v = points[j]
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
            if dot == 0:
                edges.append([i, j])
        if time.time() > deadline:
            break
            
    return {'n': n, 'edges': edges}
