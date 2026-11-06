    def can_add(adj, u, v):
        # Check if path of length 4 exists between u and v
        # u-x-y-z-w-v
        # x in N(u), w in N(v)
        # y in N(x), z in N(w)
        # y-z is an edge
        
        # Precompute N2(u) and N2(v) ? 
        # N2(u) = union of N(x) for x in N(u)
        # This is expensive to recompute for every pair.
        
        # Alternative:
        # Iterate x in N(u)
        #   Compute S_x = union of N(y) for y in N(x)  # Vertices at dist 2 from u via x
        #   Actually, we just need to know if any z in S_x is connected to any w in N(v)
        #   Wait, the edge is y-z. z is in N(w).
        #   So for a fixed x, we have the set of y's: N(x).
        #   We need to know if any y in N(x) is connected to any z that is in N(w) for some w in N(v).
        #   Let B = union of N(w) for w in N(v). This is N2(v).
        #   So we need to check if there is an edge between N(x) and N2(v).
        
        # So:
        # 1. Compute N2_v = union of N(w) for w in N(v).
        # 2. For each x in N(u):
        #      For each y in N(x):
        #          If adj[y] & N2_v != 0 (excluding u, v, x, y, and the specific w? No, just distinctness)
        
        # Distinctness check is tricky with just bitsets.
        # u, v, x, y, z, w must be distinct.
        # z is in N2_v. z is adjacent to some w in N(v).
        # If adj[y] & N2_v != 0, let z be such a vertex.
        # We need z != u, v, x, y.
        # And w (neighbor of z in N(v)) != u, v, x, y, z.
        
        # Given the small N=40, we can afford some iteration.
        
        Nv = adj[v]
        Nu = adj[u]
        
        # Compute N2_v: vertices at distance 2 from v
        # N2_v = union of N(w) for w in N(v)
        N2_v = 0
        mask = Nv
        while mask:
            w = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            N2_v |= adj[w]
        
        # Remove u and v from N2_v? 
        # z cannot be u or v.
        N2_v &= ~(1 << u)
        N2_v &= ~(1 << v)
        
        # Now check for x in N(u)
        mask_x = Nu
        while mask_x:
            x = (mask_x & -mask_x).bit_length() - 1
            mask_x &= mask_x - 1
            
            if x == v: continue # x is in N(u), if x==v edge already exists? No, we are adding (u,v). So v not in N(u) yet.
            
            # For each y in N(x)
            mask_y = adj[x]
            # y cannot be u, v, x
            mask_y &= ~(1 << u)
            mask_y &= ~(1 << v)
            mask_y &= ~(1 << x)
            
            while mask_y:
                y = (mask_y & -mask_y).bit_length() - 1
                mask_y &= mask_y - 1
                
                # Check if y has any neighbor z in N2_v
                # z must be distinct from u, v, x, y
                # N2_v already excludes u, v.
                # We need to exclude x and y from the intersection.
                # Also, the neighbor w of z in N(v) must be distinct from x, y, z, u, v.
                
                # If adj[y] & N2_v != 0, there is a candidate z.
                # Let's find z.
                candidates_z = adj[y] & N2_v
                candidates_z &= ~(1 << x)
                candidates_z &= ~(1 << y)
                
                if candidates_z:
                    # There is a z.
                    # We need to ensure there exists a w in N(v) such that w~z and w is distinct from u,v,x,y,z.
                    # z is in N2_v, so z is adjacent to at least one w in N(v).
                    # Let W_z = adj[z] & Nv.
                    # We need W_z to contain a w not in {u, v, x, y, z}.
                    # u, v are not in Nv (simple graph, no self loops, and u,v not adjacent yet).
                    # So W_z & ~(1<<x) & ~(1<<y) & ~(1<<z) must be non-empty.
                    
                    z_mask = candidates_z
                    while z_mask:
                        z = (z_mask & -z_mask).bit_length() - 1
                        z_mask &= z_mask - 1
                        
                        W_z = adj[z] & Nv
                        # w cannot be u, v (not in Nv anyway), x, y, z
                        W_z &= ~(1 << x)
                        W_z &= ~(1 << y)
                        W_z &= ~(1 << z)
                        
                        if W_z:
                            return False
        return True
    