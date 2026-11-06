import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # We want to maximize n.
    # Try n from 47 down to 40.
    # If we find a valid graph for n, return it.
    
    for n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
            
        min_deg_req = n - 39
        
        # Multiple attempts for each n
        # Increase attempts for smaller n where success is more likely
        # But for n=47, we need to be fast per attempt to try more.
        if n >= 46:
            max_attempts = 100
        else:
            max_attempts = 200
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # 1. Initialize
            adj = [0] * n
            degrees = [0] * n
            
            # Generate all possible edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # 2. Greedy construction
            # We want to build a graph that is as dense as possible while avoiding C4.
            # A good heuristic is to process edges in random order, but maybe
            # prioritize edges connecting low-degree vertices?
            # Standard greedy: add if no C4.
            for u, v in edges:
                if time.time() > deadline - 2:
                    break
                # Check if adding (u, v) creates a C4
                # Condition: No common neighbor in N(u) and N(v)
                # Wait, C4 is u-a-v-b-u.
                # If we add (u,v), we create C4 if there exists a in N(u) and b in N(v)
                # such that a-b is an edge? No.
                # C4 is a cycle of length 4.
                # If we add edge (u,v), a C4 is formed if there is a path of length 3 between u and v.
                # i.e., u-a-b-v where (u,a), (a,b), (b,v) are edges.
                # This is equivalent to: N(u) and N(v) share a common neighbor? No.
                # If a in N(u) and a in N(v), then u-a-v is a path of length 2. Adding (u,v) creates a triangle (C3), not C4.
                # C4 involves 4 distinct vertices.
                # If we add (u,v), a C4 exists if there is a vertex a such that a is in N(u) and there is a vertex b in N(v) such that (a,b) is an edge?
                # No, the cycle is u-a-b-v-u.
                # So we need a in N(u) and b in N(v) such that (a,b) is an edge.
                # Also a != v and b != u (guaranteed if simple graph and u,v not adjacent yet).
                # Also a != b (guaranteed if simple graph).
                
                c4_found = False
                # Iterate over neighbors of u
                temp = adj[u]
                while temp:
                    lsb = temp & (-temp)
                    a = lsb.bit_length() - 1
                    # Check if any neighbor of a is in N(v)
                    # If adj[a] & adj[v] is non-zero, there is an edge (a,b) with b in N(v).
                    if adj[a] & adj[v]:
                        c4_found = True
                        break
                    temp ^= lsb
                
                if not c4_found:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
            
            # 3. Local Search
            # Goal: Increase min degree to at least min_deg_req
            improved = True
            polish_deadline = deadline - 2
            while improved and time.time() < polish_deadline:
                improved = False
                min_deg = min(degrees)
                
                if min_deg >= min_deg_req:
                    break
                
                # Find vertices with min degree
                min_deg_val = min(degrees)
                low_verts = [i for i in range(n) if degrees[i] == min_deg_val]
                random.shuffle(low_verts)
                
                # Phase 1: Try to add edges to low degree vertices
                added_any = False
                for u in low_verts:
                    if time.time() > polish_deadline:
                        break
                    
                    # Candidates for v: non-neighbors of u
                    non_neighbors = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
                    random.shuffle(non_neighbors)
                    
                    for v in non_neighbors:
                        # Check C4 for adding (u, v)
                        c4_found = False
                        temp = adj[u]
                        while temp:
                            lsb = temp & (-temp)
                            a = lsb.bit_length() - 1
                            if adj[a] & adj[v]:
                                c4_found = True
                                break
                            temp ^= lsb
                        
                        if not c4_found:
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            improved = True
                            added_any = True
                            break
                    
                    if added_any:
                        break # Restart loop as degrees changed
                
                # Phase 2: If no edge could be added, we are stuck in a local optimum.
                # Instead of complex swaps, we will just break out and restart with a new random seed.
                # This is often more effective for these sparse constraints than trying to fix a stuck state.
                if not improved:
                    break

            # Final check
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
                # Valid graph found
                edges_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges_list.append([i, j])
                return {'n': n, 'edges': edges_list}
        
    # Fallback: return a trivial valid graph (e.g., n=39, empty graph? No, min deg 0. Req 0. Valid.)
    # Or n=40, min deg 1.
    # Let's return n=39, edges=[]
    return {'n': 39, 'edges': []}
