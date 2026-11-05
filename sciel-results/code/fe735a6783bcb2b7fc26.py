"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We construct a subgraph of the polarity graph of PG(2,7).
    # PG(2,7) has 57 points. The polarity graph is C4-free.
    # Its independence number is at most q+1 = 8.
    # We select 43 vertices, which is the maximum allowed n.
    
    q = 7
    points = []
    
    # Generate all points of PG(2,q)
    # Points are normalized non-zero vectors in F_q^3
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    x_n = 1
                    y_n = (y * inv_x) % q
                    z_n = (z * inv_x) % q
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    x_n = 0
                    y_n = 1
                    z_n = (z * inv_y) % q
                else:
                    # z != 0
                    inv_z = pow(z, q-2, q)
                    x_n = 0
                    y_n = 0
                    z_n = 1
                
                points.append((x_n, y_n, z_n))
    
    # Ensure uniqueness (normalization should handle it, but just in case)
    points = list(set(points))
    
    # We expect 57 points
    # Select the first 43 points
    selected_points = points[:43]
    n = len(selected_points)
    
    # Build edges
    edges = []
    for i in range(n):
        u = selected_points[i]
        for j in range(i + 1, n):
            v = selected_points[j]
            # Check orthogonality: u.x * v.x + u.y * v.y + u.z * v.z == 0 mod q
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
