import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    q = 5
    n = q * q + q + 1  # 31
    
    # Generate points of PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero is 1
                if x != 0:
                    inv_x = pow(x, -1, q)
                    points.append([1, (y * inv_x) % q, (z * inv_x) % q])
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    points.append([0, 1, (z * inv_y) % q])
                else:
                    points.append([0, 0, 1])
    
    if len(points) != n:
        # Fallback
        return [[i, j] for i in range(n) for j in range(i + 1, n)][:91]

    best_min_common = float('inf')
    best_matrix = None
    best_pair = None

    # Iterate over all symmetric matrices
    # M = [[a, b, c], [b, d, e], [c, e, f]]
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    for e in range(q):
                        for f in range(q):
                            if time.time() > deadline:
                                break
                            
                            # Calculate determinant
                            # det = a(df - e^2) - b(bf - ce) + c(be - cd)
                            det = (a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)) % q
                            
                            if det == 0:
                                continue
                                
                            # Build adjacency bitsets
                            adj = [0] * n
                            for i in range(n):
                                ui = points[i]
                                # Calculate M * ui
                                v0 = (a * ui[0] + b * ui[1] + c * ui[2]) % q
                                v1 = (b * ui[0] + d * ui[1] + e * ui[2]) % q
                                v2 = (c * ui[0] + e * ui[1] + f * ui[2]) % q
                                
                                mask = 0
                                for j in range(n):
                                    if i == j:
                                        continue
                                    uj = points[j]
                                    dot = (ui[0] * v0 + ui[1] * v1 + ui[2] * v2) % q
                                    # Wait, dot product is ui . (M uj) or (M ui) . uj?
                                    # Since M is symmetric, ui^T M uj = (M ui)^T uj = ui^T (M uj).
                                    # So we can precompute M * uj for each j?
                                    # Or just compute dot directly.
                                    # dot = (ui[0]*(a*uj[0]+b*uj[1]+c*uj[2]) + ui[1]*(b*uj[0]+d*uj[1]+e*uj[2]) + ui[2]*(c*uj[0]+e*uj[1]+f*uj[2])) % q
                                    # Let's use the precomputed v0,v1,v2 for ui? No, that's M*ui.
                                    # dot = ui . (M uj).
                                    # It is faster to precompute M*uj for all j first.
                                    pass
                            
                            # Re-do inner loop with precomputation
                            # Precompute M * points[j] for all j
                            M_points = []
                            for j in range(n):
                                uj = points[j]
                                v0 = (a * uj[0] + b * uj[1] + c * uj[2]) % q
                                v1 = (b * uj[0] + d * uj[1] + e * uj[2]) % q
                                v2 = (c * uj[0] + e * uj[1] + f * uj[2]) % q
                                M_points.append([v0, v1, v2])
                                
                            for i in range(n):
                                ui = points[i]
                                mask = 0
                                for j in range(n):
                                    if i == j:
                                        continue
                                    uj = points[j]
                                    # dot = ui . (M uj)
                                    dot = (ui[0] * M_points[j][0] + ui[1] * M_points[j][1] + ui[2] * M_points[j][2]) % q
                                    if dot == 0:
                                        mask |= (1 << j)
                                adj[i] = mask
                                
                            # Check non-edges and common neighbors
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if not (adj[i] & (1 << j)):
                                        common = (adj[i] & adj[j]).bit_count()
                                        if common < best_min_common:
                                            best_min_common = common
                                            best_matrix = (a, b, c, d, e, f)
                                            best_pair = (i, j)

    # Construct final graph
    a, b, c, d, e, f = best_matrix
    M_points = []
    for j in range(n):
        uj = points[j]
        v0 = (a * uj[0] + b * uj[1] + c * uj[2]) % q
        v1 = (b * uj[0] + d * uj[1] + e * uj[2]) % q
        v2 = (c * uj[0] + e * uj[1] + f * uj[2]) % q
        M_points.append([v0, v1, v2])
        
    edges = []
    for i in range(n):
        ui = points[i]
        for j in range(i + 1, n):
            uj = points[j]
            dot = (ui[0] * M_points[j][0] + ui[1] * M_points[j][1] + ui[2] * M_points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
                
    # Add the best edge
    edges.append([best_pair[0], best_pair[1]])
    
    return edges
