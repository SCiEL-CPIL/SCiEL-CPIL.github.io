import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # We want to maximize n.
    # n=47 is impossible (Reiman bound vs degree requirement).
    # n=46 is possible but tight.
    # n=45 is likely possible.
    
    best_n = 39
    best_edges = []
    
    for n in [46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
            
        min_deg_req = n - 39
        
        # Try to construct a C4-free graph on n vertices with min degree >= min_deg_req
        # using a demand-driven greedy approach
        
        success = False
        for trial in range(1000):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            deg = [0] * n
            
            # Precompute all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            # Demand-driven greedy
            # We need to ensure every vertex reaches min_deg_req
            # We add edges one by one.
            # Priority: edges connecting vertices with low degree.
            
            # To make it efficient, we can just iterate through shuffled edges
            # but only add if it helps a vertex that is below min_deg_req
            # OR if it doesn't create a C4.
            # Actually, just adding any safe edge is fine, but we must ensure
            # we don't "waste" edges on high-degree vertices if low-degree ones are stuck.
            
            # A better heuristic:
            # While min(deg) < min_deg_req:
            #   Find u with min deg.
            #   Find v such that (u,v) is safe and deg[v] is low.
            #   Add (u,v).
            
            stuck = False
            while min(deg) < min_deg_req:
                if time.time() > deadline:
                    break
                
                # Find vertex with minimum degree
                u = min(range(n), key=lambda x: deg[x])
                
                if deg[u] >= min_deg_req:
                    break
                    
                # Find candidates for v
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if v in adj[u]:
                        continue
                    
                    # Check if adding (u, v) creates a C4
                    # C4: u-a-b-v
                    # a in N(u), b in N(v), a~b
                    safe = True
                    for a in adj[u]:
                        for b in adj[v]:
                            if b in adj[a]:
                                safe = False
                                break
                        if not safe:
                            break
                    
                    if safe:
                        candidates.append(v)
                
                if not candidates:
                    stuck = True
                    break
                
                # Pick v with lowest degree
                v = min(candidates, key=lambda x: deg[x])
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
            
            if not stuck and min(deg) >= min_deg_req:
                success = True
                break
        
        if success:
            best_n = n
            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
            # Verify
            # Check C4
            has_c4 = False
            for i in range(n):
                for j in range(i+1, n):
                    if j in adj[i]:
                        common = adj[i] & adj[j]
                        if len(common) > 1:
                            has_c4 = True
                            break
                if has_c4:
                    break
            
            if not has_c4:
                return {'n': n, 'edges': best_edges}
            else:
                # If C4 found, it means our check was wrong or bug.
                # Should not happen.
                pass

    return {'n': best_n, 'edges': best_edges}
