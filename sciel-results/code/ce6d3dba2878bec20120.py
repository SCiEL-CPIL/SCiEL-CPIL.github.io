import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_n = 39
    best_edges = []
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg_req = n - 39
        
        # Try multiple randomizations
        for trial in range(500):
            if time.time() > deadline:
                break
                
            # Initialize adjacency matrix/list
            adj = [set() for _ in range(n)]
            
            # Potential edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    edges.append((i, j))
            random.shuffle(edges)
            
            valid = True
            # Greedy addition
            for u, v in edges:
                if time.time() > deadline:
                    valid = False
                    break
                
                # Check if adding (u, v) creates a C4
                # A C4 is u-a-b-v.
                # We need to check if there is a common neighbor of a and b?
                # No, we need to check if there exists a path of length 3.
                # Path u-a-b-v.
                # a in N(u), b in N(v), a ~ b.
                # Also a != v, b != u (guaranteed by simple graph and u!=v)
                # And a != b (guaranteed if no multi-edges)
                
                # Optimization:
                # If N(u) and N(v) have an edge between them, then C4.
                # Let A = N(u), B = N(v).
                # Check if there is an edge between A and B.
                
                # We can iterate over smaller set
                if len(adj[u]) < len(adj[v]):
                    for a in adj[u]:
                        for b in adj[v]:
                            if a == b: continue # Triangle, not C4
                            if b in adj[a]:
                                # Found C4: u-a-b-v
                                # Do not add edge
                                break
                        else:
                            continue
                        break
                    else:
                        # No C4 found
                        adj[u].add(v)
                        adj[v].add(u)
                else:
                    for b in adj[v]:
                        for a in adj[u]:
                            if a == b: continue
                            if b in adj[a]:
                                break
                        else:
                            continue
                        break
                    else:
                        adj[u].add(v)
                        adj[v].add(u)
            
            if not valid:
                continue
                
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_req:
                # Success
                best_n = n
                best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
                break # Found for this n, go to smaller n? No, we want max n.
                # Since we iterate n downwards, if we find one for n, it's the max.
                # But we might fail for n=47 and succeed for n=46.
                # So we break the trial loop, but not the n loop?
                # Actually, if we succeed for n, we can return immediately?
                # The task is to maximize n.
                # If we find a valid graph for n, it's the best possible for this n.
                # We want the max n.
                # So if we find one for n=47, we return.
                # If we fail for n=47, we try n=46.
                # So we break the trial loop and return.
                return {'n': n, 'edges': best_edges}
            
        # If we didn't find one for this n, try smaller n
        continue
    
    return {'n': best_n, 'edges': best_edges}
