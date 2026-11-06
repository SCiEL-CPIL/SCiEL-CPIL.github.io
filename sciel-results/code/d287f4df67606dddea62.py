import random
import time

def count_c4(adj):
    """
    Count number of C4s in graph represented by list of bitmasks.
    """
    n = len(adj)
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            c = bin(common).count('1')
            if c >= 2:
                c4_count += c * (c - 1) // 2
    return c4_count // 4

def min_degree(adj):
    """
    Calculate minimum degree.
    """
    n = len(adj)
    min_deg = n
    for i in range(n):
        deg = bin(adj[i]).count('1')
        if deg < min_deg:
            min_deg = deg
    return min_deg

def construct():
    start_time = time.time()
    # Leave 10s margin for safety
    deadline = start_time + 140
    
    best_n = 39
    best_edges = []
    
    # Try n from 46 down to 40
    for n in range(46, 39, -1):
        if time.time() > deadline - 5:
            break
            
        required_min_deg = n - 39
        
        # Run multiple restarts
        found = False
        # Dynamic number of restarts based on time
        time_left = deadline - 5 - time.time()
        if time_left <= 0:
            break
            
        # Estimate time per restart
        # Each step is ~1ms. 1000 steps = 1s.
        # Let's try 10 restarts of 1000 steps each = 10s.
        max_restarts = 5
        steps_per_restart = 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize random graph
            # Target edges: enough to meet min degree
            target_edges = int(n * required_min_deg / 2) + 5
            
            adj = [0] * n
            
            # Generate random edges
            # Use a set to avoid duplicates
            edge_set = set()
            attempts = 0
            while len(edge_set) < target_edges and attempts < 10000:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if u > v:
                        u, v = v, u
                    if (u, v) not in edge_set:
                        edge_set.add((u, v))
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                attempts += 1
            
            # Local Search
            for step in range(steps_per_restart):
                if time.time() > deadline - 2:
                    break
                    
                current_c4 = count_c4(adj)
                current_min_deg = min_degree(adj)
                
                if current_c4 == 0 and current_min_deg >= required_min_deg:
                    # Found a valid graph
                    best_n = n
                    final_edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                final_edges.append([i, j])
                    best_edges = final_edges
                    found = True
                    break
                    
                # If not valid, try to improve
                # Strategy: Remove an edge from a C4 if C4s exist
                if current_c4 > 0:
                    # Find a C4 and remove an edge
                    # To speed up, pick random pairs until we find one with common neighbors
                    moved = False
                    for _ in range(20):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u == v:
                            continue
                        common = adj[u] & adj[v]
                        c = bin(common).count('1')
                        if c >= 2:
                            # Get two common neighbors
                            common_list = []
                            for k in range(n):
                                if common & (1 << k):
                                    common_list.append(k)
                            # Pick two distinct
                            w = random.choice(common_list)
                            x = random.choice([k for k in common_list if k != w])
                            # Cycle u-w-v-x-u
                            # Remove edge (u, w)
                            if adj[u] & (1 << w):
                                adj[u] ^= (1 << w)
                                adj[w] ^= (1 << u)
                                moved = True
                                break
                    if not moved:
                        # If we couldn't find a C4 easily (shouldn't happen if count > 0),
                        # or if we picked a bad edge, try adding an edge if min deg is low
                        pass
                elif current_min_deg < required_min_deg:
                    # Try to add an edge to a low degree vertex
                    low_deg_vertices = [i for i in range(n) if bin(adj[i]).count('1') < required_min_deg]
                    if low_deg_vertices:
                        u = random.choice(low_deg_vertices)
                        # Find a candidate v
                        candidates = [v for v in range(n) if v != u and not (adj[u] & (1 << v))]
                        if candidates:
                            v = random.choice(candidates)
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Occasionally restart if stuck in local minimum
                if step > 500 and step % 100 == 0:
                    if count_c4(adj) > 10:
                        break

            if found:
                break
                
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
