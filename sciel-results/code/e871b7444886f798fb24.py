import random
import time

def construct():
    n = 40
    # adj[u] is a set of neighbors of u
    adj = [set() for _ in range(n)]
    
    def creates_c6(u, v):
        # Check if adding edge (u, v) creates a C6.
        # This happens if there is a simple path of length 5 from u to v in the current graph.
        # Path: u - a - b - c - d - v
        # a in N(u)
        # b in N(a), b != u
        # c in N(b), c not in {u, a}
        # d in N(c), d not in {u, a, b, c}
        # v in N(d)
        
        # Optimization: 
        # Iterate a in N(u)
        for a in adj[u]:
            if a == v: continue # Edge already exists or self loop, but we check pairs
            # Actually if (u,v) is not in graph, a != v is guaranteed if v not in N(u).
            # But we are checking before adding, so v is not in N(u).
            
            # Iterate b in N(a)
            for b in adj[a]:
                if b == u: continue
                if b == v: continue # If b==v, path is u-a-v, length 2. Adding u-v creates C3. Allowed.
                                     # We only care about C6.
                                     # Wait, if there is a path of length 5, it creates C6.
                                     # If there is a path of length 2, adding edge creates C3. That's fine.
                                     # So we only reject if a path of length 5 exists.
                                     # We don't care about shorter paths.
                                     # However, the path must be SIMPLE.
                                     # So b != u.
                                     # Also, the path u-a-b-c-d-v must have distinct vertices.
                                     # So a, b, c, d must be distinct and not u, v.
                
                # Iterate c in N(b)
                for c in adj[b]:
                    if c in (u, a, v): continue
                    # Iterate d in N(c)
                    # We need d such that d is connected to v.
                    # And d not in {u, a, b, c}
                    
                    # Check intersection of N(c) and N(v)
                    # Candidates for d are in N(c) intersect N(v)
                    # Exclude {u, a, b, c}
                    for d in adj[c]:
                        if d in (u, a, b, c, v): continue
                        if d in adj[v]:
                            # Found path u-a-b-c-d-v
                            # Vertices: u, a, b, c, d, v. All distinct?
                            # We ensured a!=u,v, b!=u,a,v, c!=u,a,b,v, d!=u,a,b,c,v.
                            # So yes, distinct.
                            return True
        return False

    best_edges = 0
    best_adj = None
    
    # Run multiple randomized greedy attempts
    start_time = time.time()
    limit = 110 # seconds
    
    for attempt in range(50):
        if time.time() - start_time > limit:
            break
            
        # Reset
        adj = [set() for _ in range(n)]
        edges = []
        
        # Random order of edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if not creates_c6(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_adj = [set(s) for s in adj]
            best_edges_list = [e for e in edges]
            
        # If we have enough time, do a local search on the best graph found so far
        if best_edges >= 100 and time.time() - start_time < limit - 10:
             # Local search: try to add edges that were skipped
             # Or swap edges
             pass

    # Return the best found
    return best_edges_list if best_edges_list else []
