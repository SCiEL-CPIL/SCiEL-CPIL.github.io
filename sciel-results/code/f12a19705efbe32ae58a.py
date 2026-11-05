import random

def construct():
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def check_q3(u, v, adj):
        # adj is a list of integers (bitmasks)
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Extract neighbor lists
        neigh_u = []
        m = mask_u
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            neigh_u.append(idx)
            m -= lsb
            
        neigh_v = []
        m = mask_v
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            neigh_v.append(idx)
            m -= lsb
            
        len_u = len(neigh_u)
        len_v = len(neigh_v)
        
        for i in range(len_u):
            x = neigh_u[i]
            for j in range(i + 1, len_u):
                y = neigh_u[j]
                
                for k in range(len_v):
                    xp = neigh_v[k]
                    for l in range(k + 1, len_v):
                        yp = neigh_v[l]
                        
                        # Check disjointness
                        if x == xp or x == yp or y == xp or y == yp:
                            continue
                            
                        # Check edges (x, xp) and (y, yp)
                        if not ((adj[x] >> xp) & 1):
                            continue
                        if not ((adj[y] >> yp) & 1):
                            continue
                        
                        # Find common neighbors z of x and y
                        common_xy = adj[x] & adj[y]
                        forbidden = (1 << u) | (1 << v) | (1 << x) | (1 << y) | (1 << xp) | (1 << yp)
                        common_xy &= ~forbidden
                        
                        if common_xy == 0:
                            continue
                            
                        # Find common neighbors zp of xp and yp
                        common_xpyp = adj[xp] & adj[yp]
                        common_xpyp &= ~forbidden
                        
                        if common_xpyp == 0:
                            continue

                        z_mask = common_xy
                        while z_mask:
                            lsb_z = z_mask & -z_mask
                            z = lsb_z.bit_length() - 1
                            z_mask -= lsb_z
                            
                            zp_mask = common_xpyp & ~(1 << z)
                            while zp_mask:
                                lsb_zp = zp_mask & -zp_mask
                                zp = lsb_zp.bit_length() - 1
                                zp_mask -= lsb_zp
                                
                                if (adj[z] >> zp) & 1:
                                    return True
        return False

    best_edges = 0
    best_graph = []
    
    # Run multiple random trials
    for _ in range(30):
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            if not check_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append([u, v])
        
        if len(current_edges) > best_edges:
            best_edges = len(current_edges)
            best_graph = current_edges
            
    return best_graph
