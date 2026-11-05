import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_valid(n, adj):
        min_deg = n - 39
        # Check min degree
        for u in range(n):
            if len(adj[u]) < min_deg:
                return False
        # Check C4: no two vertices should share 2 or more neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    # Count common neighbors
                    common = 0
                    for w in adj[u]:
                        if w in adj[v]:
                            common += 1
                            if common >= 2:
                                return False
        return True

    def find_c4(n, adj):
        # Returns (u, v, w, x) if C4 found, else None
        # A C4 is u-v-w-x-u. 
        # Equivalent to finding u, v with 2 common neighbors w, x.
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = []
                    for w in adj[u]:
                        if w in adj[v]:
                            common.append(w)
                    if len(common) >= 2:
                        w1 = common[0]
                        w2 = common[1]
                        return (u, v, w1, w2)
        return None

    def try_construct(n, seed):
        random.seed(seed)
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
            
        # Initialize with random graph
        # Target edges ~ n * min_deg / 2
        target_edges = int(n * min_deg / 2) + 5
        
        adj = [set() for _ in range(n)]
        
        # Add random edges
        added = 0
        attempts = 0
        while added < target_edges and attempts < 10000:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or v in adj[u]:
                attempts += 1
                continue
            # Check if adding u-v creates C4
            # u and v must have no common neighbors
            has_common = False
            for w in adj[u]:
                if w in adj[v]:
                    has_common = True
                    break
            if not has_common:
                adj[u].add(v)
                adj[v].add(u)
                added += 1
            attempts += 1
        
        # Local Search
        max_iter = 2000
        for i in range(max_iter):
            # Check validity
            valid = True
            # Check degrees
            for u in range(n):
                if len(adj[u]) < min_deg:
                    valid = False
                    break
            
            if valid:
                # Check C4
                c4 = find_c4(n, adj)
                if c4 is None:
                    # Valid graph found
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return edges
            
            # Repair
            c4 = find_c4(n, adj)
            if c4:
                u, v, w1, w2 = c4
                # Remove one edge from the C4
                # Prefer removing an edge that doesn't reduce degree below min
                # The cycle is u-v-w1-u and u-v-w2-u? No, u and v share w1 and w2.
                # The C4 is u-w1-v-w2-u.
                # Edges: (u,w1), (w1,v), (v,w2), (w2,u)
                # Let's pick a random edge from the C4 to remove
                edges_to_remove = [(u, w1), (w1, v), (v, w2), (w2, u)]
                # Sort by degree of endpoints (remove edge connecting high degree nodes? or low?)
                # Removing an edge reduces degree of both endpoints.
                # We want to keep degrees high.
                # If we remove (u, w1), deg(u)-1, deg(w1)-1.
                # Pick randomly to avoid bias
                random.shuffle(edges_to_remove)
                ru, rv = edges_to_remove[0]
                adj[ru].discard(rv)
                adj[rv].discard(ru)
            else:
                # No C4, but degree too low
                # Find a vertex with degree < min_deg
                low_u = -1
                for u in range(n):
                    if len(adj[u]) < min_deg:
                        low_u = u
                        break
                
                if low_u == -1:
                    # Should not happen if valid check failed on degree
                    continue
                    
                # Try to add an edge incident to low_u
                candidates = [v for v in range(n) if v != low_u and v not in adj[low_u]]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    # Check if adding (low_u, v) creates C4
                    has_common = False
                    for w in adj[low_u]:
                        if w in adj[v]:
                            has_common = True
                            break
                    if not has_common:
                        # Also check if v's degree allows it? No, adding increases degree.
                        # But we must ensure we don't create C4.
                        adj[low_u].add(v)
                        adj[v].add(low_u)
                        added = True
                        break
                if not added:
                    # Stuck, try removing a random edge to allow future additions?
                    # Or just fail this restart
                    return None
        return None

    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        found = False
        # Try multiple seeds
        for seed in range(1000):
            if time.time() > deadline:
                break
            edges = try_construct(n, seed)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
        if found:
            # If we found a graph for n, we can stop if n is max? 
            # The task says "Maximize n". 
            # Since we iterate downwards, the first valid n is the max.
            break
            
    return {'n': best_n, 'edges': best_edges}

