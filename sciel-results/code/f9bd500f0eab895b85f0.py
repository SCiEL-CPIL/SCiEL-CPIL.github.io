import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    # Try to find a graph for n=47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # Limit restarts to ensure we finish in time
        # 50 restarts should be enough if the algorithm is good
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Randomize the initial priority or edge order?
            # We can shuffle the vertices to break symmetries
            random.seed(restart * 12345 + 789)
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            edge_set = set()
            
            # Priority queue: (priority, u, v)
            # We want to add edges that increase degrees the most
            # Initial priority can be 0, or we can just push all edges
            # Pushing all edges is O(n^2 log n^2) which is fine for n=47
            
            heap = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    heapq.heappush(heap, (0, u, v))
            
            while heap:
                if time.time() > deadline:
                    break
                    
                pri, u, v = heapq.heappop(heap)
                
                # Skip if edge already exists
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                
                # Check if adding (u, v) creates a C4
                # C4 exists if there is x in N(u) such that N(x) intersects N(v)
                # Or y in N(v) such that N(y) intersects N(u)
                # Using bitmasks
                mask_u = adj[u]
                mask_v = adj[v]
                
                valid = True
                # Iterate over neighbors of u
                # To optimize, we can check the smaller set
                if bin(mask_u).count('1') > bin(mask_v).count('1'):
                    # Iterate over v's neighbors
                    m = mask_v
                    while m:
                        lsb = m & -m
                        y = lsb.bit_length() - 1
                        if adj[y] & mask_u:
                            valid = False
                            break
                        m ^= lsb
                else:
                    # Iterate over u's neighbors
                    m = mask_u
                    while m:
                        lsb = m & -m
                        x = lsb.bit_length() - 1
                        if adj[x] & mask_v:
                            valid = False
                            break
                        m ^= lsb
                
                if valid:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    
                    # Update heap with new edges involving u and v
                    # Priority is deg[u] + deg[w]
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if (u, w) not in edge_set and (w, u) not in edge_set:
                            heapq.heappush(heap, (deg[u] + deg[w], u, w))
                        if (v, w) not in edge_set and (w, v) not in edge_set:
                            heapq.heappush(heap, (deg[v] + deg[w], v, w))
            
            # Check if minimum degree requirement is met
            if min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
        
        if found:
            # If we found a solution for this target_n, we can stop
            # since we are iterating downwards
            break
            
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
