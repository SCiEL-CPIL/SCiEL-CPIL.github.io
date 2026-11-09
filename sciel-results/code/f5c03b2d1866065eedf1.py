def construct():
    q = 5
    n = q*q + q + 1
    
    # Generate points of PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    points.append((1, (y*inv_x)%q, (z*inv_x)%q))
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    points.append((0, 1, (z*inv_y)%q))
                else:
                    inv_z = pow(z, q-2, q)
                    points.append((0, 0, 1))
                    
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # We will search for a valid symmetric matrix A
    # such that the orthogonality graph is triangle-free.
    
    found = False
    edges = []
    
    # Iterate over a subset of symmetric matrices
    # To keep it fast, we try a few structured ones first, then a small random-like search
    # But deterministic is required.
    
    for a11 in range(1, q):
        if found: break
        for a22 in range(1, q):
            if found: break
            for a33 in range(1, q):
                if found: break
                for a12 in range(q):
                    if found: break
                    for a13 in range(q):
                        if found: break
                        for a23 in range(q):
                            if found: break
                            A = [[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]]
                            
                            # Check non-singular
                            a, b, c = A[0]
                            d, e, f = A[1]
                            g, h, i = A[2]
                            det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)) % q
                            if det == 0:
                                continue
                                
                            adj = [[False]*n for _ in range(n)]
                            cand_edges = []
                            
                            for i_idx in range(n):
                                for j_idx in range(i_idx+1, n):
                                    u = points[i_idx]
                                    v = points[j_idx]
                                    dot = 0
                                    for k in range(3):
                                        for l in range(3):
                                            dot = (dot + u[k] * A[k][l] * v[l]) % q
                                    if dot == 0:
                                        cand_edges.append([i_idx, j_idx])
                                        adj[i_idx][j_idx] = True
                                        adj[j_idx][i_idx] = True
                                        
                            # Check triangle-free
                            tri_free = True
                            for k in range(n):
                                # Neighbors of k
                                # Since degree is small, we can just iterate
                                for u in range(n):
                                    if not adj[k][u]: continue
                                    for v in range(u+1, n):
                                        if adj[k][v] and adj[u][v]:
                                            tri_free = False
                                            break
                                    if not tri_free: break
                                if not tri_free: break
                            
                            if tri_free:
                                found = True
                                edges = cand_edges
                                break
                                
    if found:
        return {'n': n, 'edges': edges}
        
    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
