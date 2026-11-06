<<<<<<< SEARCH
                # Pick a random candidate u to avoid bias
                u = random.choice(candidates_u)
                
                # Find safe neighbors v for u
                safe_vs = []
                mask_u = adj[u]
                
                # Iterate through all v != u
                # We can iterate through bits of ~mask_u (excluding u)
                # But n is small, so just loop v in range(n)
                
                for v in range(n):
                    if v == u:
                        continue
                    if (mask_u >> v) & 1:
                        continue # Already connected
                    
                    # Check if adding (u, v) creates a C4.
                    # A C4 is u-a-b-v.
                    # a is in N(u), b is in N(v), and a~b.
                    # So we need to check if there exists a in N(u) such that
                    # N(a) intersects N(v).
                    # N(v) is adj[v].
                    # N(a) is adj[a].
                    # We need: (adj[u] & adj[v] & adj[a]) != 0 for some a in N(u)?
                    # No.
                    # The cycle is u - a - b - v - u.
                    # Edges: (u,a), (a,b), (b,v), (v,u).
                    # We are adding (u,v).
                    # We need to check if there is a path u-a-b-v of length 3.
                    # a is a neighbor of u.
                    # b is a neighbor of v.
                    # a and b are adjacent.
                    
                    # So:
                    # For each a in N(u):
                    #   If (adj[a] & adj[v]) != 0:
                    #       Then there is a b in N(a) that is also in N(v).
                    #       This forms a C4.
                    
                    is_safe = True
                    # Iterate through neighbors of u
                    mask_a_iter = mask_u
                    while mask_a_iter:
                        # Get lowest set bit
                        lsb = mask_a_iter & -mask_a_iter
                        a_idx = lsb.bit_length() - 1
                        mask_a_iter ^= lsb
                        
                        # Check if N(a) intersects N(v)
                        if adj[a_idx] & adj[v]:
                            is_safe = False
                            break
                    
                    if is_safe:
                        safe_vs.append(v)
                
                if not safe_vs:
                    stuck = True
                    break
                
                # Pick v
                # Heuristic: Pick v with lowest degree to balance
                # Or random?
                # Let's try a mix: 50% lowest degree, 50% random
                if random.random() < 0.5:
                    v = min(safe_vs, key=lambda x: deg[x])
                else:
                    v = random.choice(safe_vs)
=======
                # Heuristic: Pick u with minimum degree to balance the graph
                u = min(candidates_u, key=lambda x: deg[x])
                
                # Find safe neighbors v for u
                safe_vs = []
                mask_u = adj[u]
                
                # Iterate through all v != u
                for v in range(n):
                    if v == u:
                        continue
                    if (mask_u >> v) & 1:
                        continue # Already connected
                    
                    # Check if adding (u, v) creates a C4.
                    is_safe = True
                    # Iterate through neighbors of u
                    mask_a_iter = mask_u
                    while mask_a_iter:
                        # Get lowest set bit
                        lsb = mask_a_iter & -mask_a_iter
                        a_idx = lsb.bit_length() - 1
                        mask_a_iter ^= lsb
                        
                        # Check if N(a) intersects N(v)
                        if adj[a_idx] & adj[v]:
                            is_safe = False
                            break
                    
                    if is_safe:
                        safe_vs.append(v)
                
                if not safe_vs:
                    stuck = True
                    break
                
                # Pick v
                # Heuristic: Pick v with lowest degree to balance
                v = min(safe_vs, key=lambda x: deg[x])
>>>>>>> REPLACE

This change makes the heuristic deterministic (Min-Min). If it gets stuck frequently, we rely on restarts. To improve restart efficiency, we can also try to vary the vertex ordering slightly or use a "randomized Min-Min" where we pick from the top 2-3 min-degree vertices.

Let's add a small amount of randomness to avoid getting stuck in the same local minima repeatedly. Instead of always picking the absolute min, we can pick randomly from the vertices with degree within 1 of the minimum.

