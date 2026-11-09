import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n from 47 down to 1
    for n in range(47, 0, -1):
        if time.time() > deadline:
            break
            
        # Required min degree
        req_deg = n - 39
        
        # If req_deg <= 0, empty graph works
        if req_deg <= 0:
            return {'n': n, 'edges': []}
            
        # Try to construct graph
        # We will use a greedy randomized approach
        # Multiple restarts if possible
        
        best_edges = []
        for restart in range(10):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            edges = []
            
            # Generate all possible edges
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                if time.time() > deadline:
                    break
                    
                # Check if adding (u,v) creates C4
                is_c4 = False
                # Check if any neighbor of u is connected to any neighbor of v
                # Note: if w in N(u) and w in N(v), it's a triangle, which is fine.
                # We only care about w in N(u), x in N(v) with w != x and w~x.
                
                # Optimization: if len(N(u)) * len(N(v)) is large, it might be slow.
                # But n is small.
                
                for w in adj[u]:
                    if w == v: continue
                    for x in adj[v]:
                        if x == u: continue
                        if w == x:
                            continue
                        if x in adj[w]:
                            is_c4 = True
                            break
                    if is_c4:
                        break
                
                if not is_c4:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= req_deg:
                return {'n': n, 'edges': edges}
            
            if len(edges) > len(best_edges):
                best_edges = edges

    # If no valid graph found for n > 39, return the best effort or empty graph for 39
    # The seed program returns 39.
    # If we found a graph for some n < 47, we should return it.
    # But the loop breaks on time.
    # If we didn't return in the loop, we might have best_edges for some n.
    # However, the logic above returns immediately if valid.
    # If it finishes all restarts for a specific n without success, it moves to n-1.
    
    # Fallback
    return {'n': 39, 'edges': []}
