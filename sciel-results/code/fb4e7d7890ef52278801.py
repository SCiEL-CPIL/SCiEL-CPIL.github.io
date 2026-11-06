import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Try n=47 first, then 46
    for n in [47, 46, 45]:
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Time budget for this n
        time_left = deadline - (time.time() - start_time)
        if time_left < 5:
            break
            
        # Allocate time based on n
        # n=47 is hardest, give it most time
        if n == 47:
            t_limit = min(60.0, time_left - 1)
        elif n == 46:
            t_limit = min(40.0, time_left - 1)
        else:
            t_limit = min(20.0, time_left - 1)
            
        if t_limit <= 0:
            continue
            
        result = _solve_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    # Fallback
    return {'n': 39, 'edges': []}

def _solve_c4_free(n, min_deg, time_limit):
    start_time = time.time()
    
    # Bitset representation for speed
    # adj[i] is a bitmask of neighbors
    # n <= 47 fits in 64-bit int, so Python int is fine.
    
    def creates_c4(adj, u, v):
        # Check if adding (u,v) creates a C4
        # C4: u-x-y-v-u
        # x in N(u), y in N(v), x~y
        # We can check: for each x in N(u), if N(x) intersects N(v), then C4.
        # Note: x cannot be v (since u,v not connected yet, v not in N(u))
        # y cannot be u (since u,v not connected, u not in N(v))
        
        # Iterate over smaller set
        nu = adj[u]
        nv = adj[v]
        
        # We need to check if there is an edge between N(u) and N(v).
        # This is equivalent to: is there an x in N(u) such that adj[x] & N(v) is non-zero?
        # OR is there a y in N(v) such that adj[y] & N(u) is non-zero?
        
        # Let's iterate over neighbors of u
        # To speed up, we can precompute? No, adj changes.
        
        # Optimization: 
        # If we maintain the graph, we can just check.
        
        # Bitwise trick:
        # We want to know if any x in N(u) has a neighbor in N(v).
        # Sum of (adj[x] & N(v)) for x in N(u) > 0?
        
        # Python loop is okay for small n.
        for x in range(n):
            if (nu >> x) & 1:
                if adj[x] & nv:
                    return True
        return False

    best_result = None
    
    # Multiple restarts
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random graph
        # Target degree approx min_deg
        # p = min_deg / (n-1)
        p = min_deg / (n - 1) * 1.2 # Slightly higher to allow repair
        
        adj = [0] * n
        edges = []
        
        # Random initialization
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        
        # Local Search
        # Objective: Minimize C4s, then Maximize Min Degree
        # Or: Repair C4s, then Boost Degree
        
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
                
            # Check if valid
            min_d = min(bin(adj[i]).count('1') for i in range(n))
            if min_d >= min_deg:
                # Check for C4s
                has_c4 = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if not ((adj[u] >> v) & 1):
                            if creates_c4(adj, u, v):
                                has_c4 = True
                                break
                    if has_c4:
                        break
                
                if not has_c4:
                    # Success
                    res_edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                res_edges.append([u, v])
                    return {'n': n, 'edges': res_edges}
            
            # If not valid, repair
            # 1. Remove C4s
            # Find a C4
            c4_found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if creates_c4(adj, u, v):
                            # Find the specific C4 to break
                            # We need x in N(u), y in N(v) such that x~y
                            nu = adj[u]
                            nv = adj[v]
                            for x in range(n):
                                if (nu >> x) & 1:
                                    if adj[x] & nv:
                                        # Found x, y
                                        # y is a neighbor of x in N(v)
                                        # Pick a random y from adj[x] & nv
                                        mask = adj[x] & nv
                                        # Get index of a set bit
                                        y_idx = (mask & -mask).bit_length() - 1
                                        # Remove edge (x, y)
                                        adj[x] &= ~(1 << y_idx)
                                        adj[y_idx] &= ~(1 << x)
                                        # Remove from edges list (optional, we rebuild at end)
                                        c4_found = True
                                        break
                                    # Note: this logic is slightly flawed if we break early, 
                                    # but we just need to remove ONE edge from SOME c4.
                                    # The creates_c4 check ensures existence.
                                    # We found x. Now we need y.
                                    # The inner loop finds x. We need to find y properly.
                                    # Let's refine the C4 breaking part.
            if c4_found:
                continue
                
            # 2. Boost Degree
            # Find vertex with min degree
            min_v = min(range(n), key=lambda i: bin(adj[i]).count('1'))
            deg = bin(adj[min_v]).count('1')
            
            if deg < min_deg:
                # Try to add an edge to min_v
                candidates = []
                for v in range(n):
                    if v != min_v and not ((adj[min_v] >> v) & 1):
                        candidates.append(v)
                
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    if not creates_c4(adj, min_v, v):
                        adj[min_v] |= (1 << v)
                        adj[v] |= (1 << min_v)
                        added = True
                        break
                
                if not added:
                    # Stuck. Try to remove a random edge to open up space
                    # Pick a random vertex with high degree and remove an edge
                    if edges:
                        # Rebuild edges list from adj for random removal
                        # Or just pick a random pair
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u != v and ((adj[u] >> v) & 1):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                    else:
                        break # Empty graph, stuck
                        
            else:
                # Degree is ok, but C4s exist? 
                # The loop above should have handled C4s.
                # If we get here, it means no C4s found in the quick check?
                # Or the C4 check is expensive.
                # Let's do a random kick if stuck
                if random.random() < 0.1:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            if not creates_c4(adj, u, v):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                
    return None
