import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    # Precompute all edges with their bit index
    edges = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v, i))
    
    best_count = 0
    best_edges = []
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # First, run the deterministic order to guarantee at least the seed's performance
    # The seed order is x in range, i in range.
    # My edges list is built similarly.
    # So the first iteration without shuffle is the seed's behavior.
    
    for trial in range(10000): # Upper bound on trials, will break on time
        if time.time() > deadline:
            break
        
        if trial > 0:
            random.shuffle(edges)
        
        chosen = set()
        current_edges = []
        
        for u, v, i in edges:
            ok = True
            # Check for C4
            # u-v is edge i.
            # C4 exists if for some j != i:
            # (u, u^2^j) in E, (v, v^2^j) in E, (u^2^j, v^2^j) in E
            for j in range(d):
                if j == i:
                    continue
                uj = u ^ (1 << j)
                vj = v ^ (1 << j)
                
                # Check edges
                e1 = (u, uj) if u < uj else (uj, u)
                if e1 not in chosen:
                    continue
                e2 = (v, vj) if v < vj else (vj, v)
                if e2 not in chosen:
                    continue
                e3 = (uj, vj) if uj < vj else (vj, uj)
                if e3 not in chosen:
                    continue
                
                # All three exist, so adding (u,v) creates a C4
                ok = False
                break
            
            if ok:
                e = (u, v) if u < v else (v, u)
                chosen.add(e)
                current_edges.append([u, v])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            if best_count >= 308:
                break
                
    return best_edges
