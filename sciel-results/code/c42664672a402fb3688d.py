def construct():
    # Polarity graph of PG(2, 5)
    q = 5
    n = q * q + q + 1
    # Points are triples (x, y, z) not all zero, modulo scaling
    # We can represent them as (x, y) with z=1, and (0, 0) for infinity?
    # Standard representation:
    # 25 affine points (x, y)
    # 5 points at infinity (slope m, and vertical)
    
    # Actually, easier:
    # Generate all points in PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Polarity: (x, y, z) is adjacent to (u, v, w) if xw + yv + zu = 0 (symmetric)
    # Wait, the polarity is usually xz + yw = 0? 
    # For PG(2, q), a symmetric polarity is given by the bilinear form xz + yw + (x+w)z? 
    # Standard: xz + yw = 0 is not symmetric in all terms? 
    # The form is $x_1 x_2 + y_1 y_2 + z_1 z_2 = 0$? No, that's orthogonal.
    # The standard polarity graph uses the form $x z + y w = 0$? 
    # Actually, the graph is defined by $P$ adjacent to $Q$ if $P$ lies on the polar line of $Q$.
    # The polar line of $(u, v, w)$ is $u X + v Y + w Z = 0$.
    # So $(x, y, z)$ is adjacent to $(u, v, w)$ if $u x + v y + w z = 0$.
    # This is symmetric if the form is symmetric.
    
    edges = []
    for i in range(n):
        p = points[i]
        for j in range(i + 1, n):
            q = points[j]
            # Check if p lies on polar of q
            # u*x + v*y + w*z = 0
            if (q[0] * p[0] + q[1] * p[1] + q[2] * p[2]) % q == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
