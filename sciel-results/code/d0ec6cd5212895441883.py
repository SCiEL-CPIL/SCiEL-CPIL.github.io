    def heuristic_triangle_free_size(adj):
        # Greedy: remove vertices involved in triangles
        # To make it deterministic-ish but varied, we can randomize the order of triangles
        # Or just use a fixed order.
        
        # For speed, just check if the whole set is triangle free? No.
        # Greedy removal:
        # Copy adj? No, we just need the size.
        # We can simulate the removal.
        # active = set(range(n))
        # while True:
        #    find a triangle in active
        #    if none, return len(active)
        #    remove a vertex from the triangle
        
        # This is O(n^3) per call?
        # Finding a triangle is O(n^3) worst case.
        # But n=18 is small.
        
        active_mask = (1 << n) - 1
        adj_mask = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    
        # Find triangles in active_mask
        # We can precompute triangles for the graph, but the graph changes.
        # So we must recompute.
        
        # Fast triangle check for a mask:
        # For each u in mask, for each v in mask with u<v, check if common neighbor in mask.
        
        while True:
            found = False
            # Iterate vertices
            m = active_mask
            while m and not found:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                
                # Neighbors of u in active_mask
                nu = adj_mask[u] & active_mask
                
                # Check for edge in nu
                nm = nu
                while nm and not found:
                    lsb2 = nm & (-nm)
                    v = lsb2.bit_length() - 1
                    nm ^= lsb2
                    
                    # Check if v is connected to any other in nu
                    # adj_mask[v] & nu
                    # If non-zero, there is a triangle (u, v, w)
                    if adj_mask[v] & nu:
                        found = True
                        # Remove u (or v or w)
                        # Random choice among u, v, w?
                        # Just remove u for simplicity
                        active_mask &= ~(1 << u)
            
            if not found:
                return active_mask.bit_count()
