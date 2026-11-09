import random

def construct():
    n = 20
    best_edges = 0
    best_edge_list = []
    
    # Run multiple restarts
    for seed in range(100):
        random.seed(seed)
        adj = [0] * n
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            valid = True
            
            # Case 3: 3 vertices in C = N(u) & N(v) have a common neighbor w != u, v
            C = adj[u] & adj[v]
            c_list = [i for i in range(n) if C & (1 << i)]
            
            # Check if any 3 in c_list have a common neighbor
            # We can just check if the intersection of neighbors of any 3 is non-empty (excluding u, v)
            # Actually, if they have a common neighbor w, then w is in N(x) & N(y) & N(z).
            # Since x,y,z are in C, they are neighbors of u and v.
            # So u and v are common neighbors.
            # We need a 3rd one.
            
            # Optimization: If |C| < 3, skip.
            if len(c_list) >= 3:
                # For each pair x, y in C, find common neighbors in C? No.
                # Find common neighbors of x and y.
                # If that set intersects with N(z) for some z in C, and contains someone other than u, v...
                # Simpler:
                # For each x, y, z in C (distinct):
                #   common = N(x) & N(y) & N(z)
                #   if common & ~(1<<u) & ~(1<<v) != 0: valid = False
                for i in range(len(c_list)):
                    x = c_list[i]
                    for j in range(i+1, len(c_list)):
                        y = c_list[j]
                        common_xy = adj[x] & adj[y]
                        # Remove u and v
                        common_xy &= ~(1 << u)
                        common_xy &= ~(1 << v)
                        # Check if any z in C (z != x, y) is in common_xy?
                        # No, z must be a neighbor of w.
                        # We need w such that w ~ x, w ~ y, w ~ z.
                        # So w is in N(x) & N(y) & N(z).
                        # Iterate z
                        for k in range(j+1, len(c_list)):
                            z = c_list[k]
                            if common_xy & adj[z]:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
            
            if valid:
                # Case 1: 2 vertices x, y in N(v) \ C have 2 common neighbors in N(u) \ {v}
                # x, y are not in N(u).
                # So x, y in N(v) & ~N(u).
                A = adj[v] & ~adj[u] & ~(1 << u) & ~(1 << v)
                a_list = [i for i in range(n) if A & (1 << i)]
                
                if len(a_list) >= 2:
                    for i in range(len(a_list)):
                        x = a_list[i]
                        for j in range(i+1, len(a_list)):
                            y = a_list[j]
                            # Common neighbors of x and y in N(u)
                            common_xy_u = adj[x] & adj[y] & adj[u]
                            # Exclude v
                            common_xy_u &= ~(1 << v)
                            # Need at least 2
                            if bin(common_xy_u).count('1') >= 2:
                                valid = False
                                break
                        if not valid: break
            
            if valid:
                # Case 2: Symmetric
                B = adj[u] & ~adj[v] & ~(1 << u) & ~(1 << v)
                b_list = [i for i in range(n) if B & (1 << i)]
                
                if len(b_list) >= 2:
                    for i in range(len(b_list)):
                        x = b_list[i]
                        for j in range(i+1, len(b_list)):
                            y = b_list[j]
                            common_xy_v = adj[x] & adj[y] & adj[v]
                            common_xy_v &= ~(1 << u)
                            if bin(common_xy_v).count('1') >= 2:
                                valid = False
                                break
                        if not valid: break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_edge_list = edges.copy()
            
    return best_edge_list
