import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_n = 39
    best_edges = []
    
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 15:
            break
            
        success = False
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            n = target_n
            min_deg_req = n - 39
            
            # Bitmask representation for fast C4 detection
            # adj[i] is a bitmask of neighbors of vertex i
            adj = [0] * n
            
            # Build edge list in random order, but weight toward edges incident to low-degree vertices
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            np.random.shuffle(all_edges)
            
            # Greedy addition with bitmask C4 check
            for u, v in all_edges:
                if time.time() > deadline - 3:
                    break
                
                # C4 check: adding (u,v) creates C4 if exists w in N(u), x in N(v) with edge (w,x)
                # Using bitmasks: for each w in N(u), check if adj[w] & adj[v] != 0
                # But we need w != v and x != u, which is automatic since (u,v) not yet in graph
                # Actually we need: exists w in N(u) such that (adj[w] & adj[v]) has a bit set for some x != u
                # Since v not in N(u) yet, and u not in N(v) yet, this is fine.
                
                # Efficient check using bitmasks
                au = adj[u]
                av = adj[v]
                
                if au == 0 or av == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
                
                # Check if any neighbor of u has an edge to any neighbor of v
                # Iterate over set bits in au
                c4_found = False
                temp = au
                while temp:
                    # Extract lowest set bit
                    w = (temp & (-temp)).bit_length() - 1
                    # Check if adj[w] shares any bit with av
                    if adj[w] & av:
                        c4_found = True
                        break
                    temp &= temp - 1  # Clear lowest set bit
                
                if not c4_found:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Compute degrees
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            min_deg = min(degrees)
            
            # Local search polish: try to improve minimum degree by edge swaps
            if min_deg < min_deg_req:
                improved = True
                polish_deadline = deadline - 2
                while improved and time.time() < polish_deadline:
                    improved = False
                    min_deg = min(degrees)
                    if min_deg >= min_deg_req:
                        break
                    
                    # Find vertices with minimum degree
                    min_deg_val = min(degrees)
                    low_verts = [i for i in range(n) if degrees[i] == min_deg_val]
                    
                    # Try to add an edge incident to a low-degree vertex
                    for u in low_verts:
                        if time.time() > polish_deadline:
                            break
                        # Try adding edge (u, v) for random v
                        candidates = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
                        np.random.shuffle(candidates)
                        for v in candidates:
                            # C4 check
                            au = adj[u]
                            av = adj[v]
                            if au == 0 or av == 0:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                degrees[u] += 1
                                degrees[v] += 1
                                improved = True
                                break
                            
                            c4_found = False
                            temp = au
                            while temp:
                                w = (temp & (-temp)).bit_length() - 1
                                if adj[w] & av:
                                    c4_found = True
                                    break
                                temp &= temp - 1
                            
                            if not c4_found:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                degrees[u] += 1
                                degrees[v] += 1
                                improved = True
                                break
                        if improved:
                            break
                    if time.time() > polish_deadline:
                        break
                
                min_deg = min(degrees)
            
            if min_deg >= min_deg_req:
                # Valid graph found
                success = True
                best_n = n
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                best_edges = edges
                break
        
        if success:
            break
    
    if best_n == 39:
        return {'n': 39, 'edges': []}
    else:
        return {'n': best_n, 'edges': best_edges}
