import random
import time

def construct():
    n = 40
    best_edges = 0
    best_graph = []
    start_time = time.time()
    time_limit = 110.0

    def has_c6(u, v, adj):
        # adj is list of ints
        nu = adj[u]
        nv = adj[v]
        
        # Path u-a-b-c-d-v
        # a in N(u), a != v
        a_mask = nu & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask -= a_bit
            
            # b in N(a), b != u, b != a, b != v
            na = adj[a]
            b_mask = na & ~(1 << u) & ~(1 << a) & ~(1 << v)
            
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask -= b_bit
                
                # c in N(b), c != a, c != b, c != u, c != v
                nb = adj[b]
                c_mask = nb & ~(1 << a) & ~(1 << b) & ~(1 << u) & ~(1 << v)
                
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask -= c_bit
                    
                    # d in N(c) and d in N(v)
                    # d != c, d != b, d != a, d != u, d != v
                    nc = adj[c]
                    d_mask = nc & nv
                    d_mask &= ~(1 << c) & ~(1 << b) & ~(1 << a) & ~(1 << u) & ~(1 << v)
                    
                    if d_mask:
                        return True
        return False

    num_runs = 1000 # Limit by time
    
    for _ in range(num_runs):
        if time.time() - start_time > time_limit:
            break
            
        adj = [0] * n
        graph = []
        candidates = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if not has_c6(u, v, adj):
                graph.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        if len(graph) > best_edges:
            best_edges = len(graph)
            best_graph = [e[:] for e in graph]
            
            if best_edges >= 105:
                break

    return best_graph
