import time

def construct():
    start_time = time.time()
    
    # We aim for n=26. If we can't, we try smaller n.
    # The known bound is 26.
    
    def try_build(n, max_time=110):
        # adj[c][v] is an integer bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(5)]
        # colors[i][j] for i < j
        colors = [[0] * n for _ in range(n)]
        
        def backtrack(v):
            if time.time() - start_time > max_time:
                return False
            
            if v == n:
                return True
            
            # Assign colors to edges (0,v), (1,v), ..., (v-1,v)
            # We need to partition {0, ..., v-1} into 5 sets S_0..S_4
            # such that for each c, S_c is an independent set in H_c
            # where H_c has edge (u, u') if adj[c][u] & adj[c][u'] != 0
            
            # To make this fast, we can iterate u from 0 to v-1 and assign colors.
            # We maintain the current assignment for u < current_u.
            
            # Optimization: Precompute conflict masks for the current v?
            # For each color c, we want to know which u's are compatible with each other.
            # Let's just use backtracking with bitmasks.
            
            def assign_edge(u_idx):
                if u_idx == v:
                    return backtrack(v + 1)
                
                # Try each color
                # Heuristic: Try colors that are less constrained?
                # For now, try 0..4
                
                for c in range(5):
                    # Check if u_idx can be added to S_c
                    # S_c currently contains u' < u_idx with colors[u'][v] == c
                    # We need adj[c][u_idx] to be disjoint from adj[c][u'] for all such u'
                    
                    # We can maintain a bitmask 'used_mask[c]' for the current v?
                    # No, the constraint is pairwise disjointness of N_c(u).
                    # If u1 and u2 are in S_c, N_c(u1) & N_c(u2) == 0.
                    # This is NOT equivalent to N_c(u_new) & (union of N_c(u_old)) == 0?
                    # Yes, it is. If N_c(u_new) intersects N_c(u1) OR N_c(u2), it fails.
                    # So we just need to check against the UNION of neighbors of all u in S_c?
                    # Wait. If N_c(u1) and N_c(u2) are disjoint, then N_c(u_new) must be disjoint from both.
                    # So N_c(u_new) & (N_c(u1) | N_c(u2)) == 0.
                    # So we can maintain `union_neigh[c]` for the current v.
                    
                    # Let's modify the scope to pass union_neigh
                    # This requires restructuring the recursion to carry state.
                    
                    # For simplicity and speed, I will implement the check directly 
                    # but using bitmasks. The number of u's is small (max 25).
                    
                    valid = True
                    # Check against all u' < u_idx that have color c
                    for u2 in range(u_idx):
                        if colors[u2][v] == c:
                            if adj[c][u2] & adj[c][u_idx]:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Assign
                    colors[u_idx][v] = c
                    colors[v][u_idx] = c
                    adj[c][u_idx] |= (1 << v)
                    adj[c][v] |= (1 << u_idx)
                    
                    if assign_edge(u_idx + 1):
                        return True
                    
                    # Unassign
                    adj[c][u_idx] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u_idx)
                    colors[u_idx][v] = 0
                    colors[v][u_idx] = 0
                    
                return False
            
            return assign_edge(0)
        
        return backtrack(1), adj, colors

    for n in range(26, 5, -1):
        if time.time() - start_time > 115:
            break
        found, adj, colors = try_build(n)
        if found:
            color_list = []
            for i in range(n):
                for j in range(i+1, n):
                    color_list.append(colors[i][j])
            return {'n': n, 'colors': color_list}
            
    return {'n': 2, 'colors': [0]}
