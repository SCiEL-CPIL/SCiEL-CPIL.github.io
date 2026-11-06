import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def has_q3(adj):
        for v0 in range(n):
            neigh_v0 = []
            mask = adj[v0]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if v != v0:
                    neigh_v0.append(v)
            
            for v1, v2, v4 in combinations(neigh_v0, 3):
                # Pruning: v1, v2, v4 must be independent in Q3
                if (adj[v1] >> v2) & 1: continue
                if (adj[v1] >> v4) & 1: continue
                if (adj[v2] >> v4) & 1: continue
                
                common_12 = adj[v1] & adj[v2]
                common_14 = adj[v1] & adj[v4]
                common_24 = adj[v2] & adj[v4]
                
                exclude_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v4)
                common_12 &= ~exclude_mask
                common_14 &= ~exclude_mask
                common_24 &= ~exclude_mask
                
                mask3 = common_12
                while mask3:
                    v3 = (mask3 & -mask3).bit_length() - 1
                    mask3 &= mask3 - 1
                    
                    mask5 = common_14
                    while mask5:
                        v5 = (mask5 & -mask5).bit_length() - 1
                        mask5 &= mask5 - 1
                        
                        if v3 == v5: continue
                        
                        mask6 = common_24
                        while mask6:
                            v6 = (mask6 & -mask6).bit_length() - 1
                            mask6 &= mask6 - 1
                            
                            if v6 == v3 or v6 == v5: continue
                            
                            common_356 = adj[v3] & adj[v5] & adj[v6]
                            exclude_mask_7 = exclude_mask | (1 << v3) | (1 << v5) | (1 << v6)
                            common_356 &= ~exclude_mask_7
                            
                            if common_356:
                                return True
        return False

    def local_search(adj):
        # Try to add edges greedily
        improved = True
        while improved:
            improved = False
            # Shuffle edges to avoid bias
            edges_list = all_edges[:]
            random.shuffle(edges_list)
            for u, v in edges_list:
                if not ((adj[u] >> v) & 1):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if has_q3(adj):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        improved = True
        return adj

    best_adj = None
    best_count = 0
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(2000 + restart)
        
        # Start with a random graph
        # Probability 0.5 is a good start
        adj = [0] * n
        for u, v in all_edges:
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Remove Q3s to make it valid
        # We can just remove random edges from Q3s until valid
        # Or use the "delete" method
        while has_q3(adj):
            # Find a Q3 and remove an edge from it
            # We need a function to find a Q3 edge, not just check existence
            # For now, just remove a random edge until valid? Too slow.
            # Let's use a simple repair: pick random edge, if removing it helps, do it?
            # Better: Use the find_q3_edges from original? 
            # Let's implement a quick find_q3_edge
            q3_found = False
            for v0 in range(n):
                neigh_v0 = []
                mask = adj[v0]
                while mask:
                    v = (mask & -mask).bit_length() - 1
                    mask &= mask - 1
                    if v != v0:
                        neigh_v0.append(v)
                
                for v1, v2, v4 in combinations(neigh_v0, 3):
                    if (adj[v1] >> v2) & 1: continue
                    if (adj[v1] >> v4) & 1: continue
                    if (adj[v2] >> v4) & 1: continue
                    
                    common_12 = adj[v1] & adj[v2]
                    common_14 = adj[v1] & adj[v4]
                    common_24 = adj[v2] & adj[v4]
                    
                    exclude_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v4)
                    common_12 &= ~exclude_mask
                    common_14 &= ~exclude_mask
                    common_24 &= ~exclude_mask
                    
                    mask3 = common_12
                    while mask3:
                        v3 = (mask3 & -mask3).bit_length() - 1
                        mask3 &= mask3 - 1
                        
                        mask5 = common_14
                        while mask5:
                            v5 = (mask5 & -mask5).bit_length() - 1
                            mask5 &= mask5 - 1
                            
                            if v3 == v5: continue
                            
                            mask6 = common_24
                            while mask6:
                                v6 = (mask6 & -mask6).bit_length() - 1
                                mask6 &= mask6 - 1
                                
                                if v6 == v3 or v6 == v5: continue
                                
                                common_356 = adj[v3] & adj[v5] & adj[v6]
                                exclude_mask_7 = exclude_mask | (1 << v3) | (1 << v5) | (1 << v6)
                                common_356 &= ~exclude_mask_7
                                
                                if common_356:
                                    # Found Q3 with edges (v0,v1), (v0,v2), (v0,v4), (v1,v3), (v1,v5), (v2,v3), (v2,v6), (v3,v7?), (v4,v5), (v4,v6), (v5,v7?), (v6,v7?)
                                    # Wait, v7 is in common_356.
                                    # We just need to break.
                                    # Let's remove edge (v0, v1)
                                    adj[v0] &= ~(1 << v1)
                                    adj[v1] &= ~(1 << v0)
                                    q3_found = True
                                    break
                        if q3_found: break
                    if q3_found: break
                if q3_found: break
            if not q3_found:
                # Should not happen if has_q3 is correct, but just in case
                break
        
        # Now adj is Q3-free. Perform Local Search
        adj = local_search(adj)
        
        ec = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        if ec > best_count:
            best_count = ec
            best_adj = adj[:]
            
        # Iterated Local Search: Perturb and search again
        # Do this a few times per restart
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            # Perturb: remove 3 random edges
            current_adj = best_adj[:]
            for _ in range(3):
                u, v = random.choice(all_edges)
                if (current_adj[u] >> v) & 1:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
            
            current_adj = local_search(current_adj)
            ec = sum(bin(current_adj[i]).count('1') for i in range(n)) // 2
            if ec > best_count:
                best_count = ec
                best_adj = current_adj[:]

    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
