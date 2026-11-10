    def can_add(adj, u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Common neighbors of u and v
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
            
        # 2. For all w in N(v) \ {u}, (u, w) gains v as common neighbor
        #    So |N(u) \cap N(w)| must be <= 1
        #    Note: if w == u, skip.
        #    w is in N(v) means (v, w) is edge.
        nv_mask = adj[v] & ~(1 << u) # Exclude u
        w = 0
        while nv_mask:
            lsb = nv_mask & -nv_mask
            w = lsb.bit_length() - 1
            nv_mask ^= lsb
            if (adj[u] & adj[w]).bit_count() > 1:
                return False
                
        # 3. For all w in N(u) \ {v}, (v, w) gains u as common neighbor
        #    So |N(v) \cap N(w)| must be <= 1
        nu_mask = adj[u] & ~(1 << v) # Exclude v
        w = 0
        while nu_mask:
            lsb = nu_mask & -nu_mask
            w = lsb.bit_length() - 1
            nu_mask ^= lsb
            if (adj[v] & adj[w]).bit_count() > 1:
                return False
                
        return True
