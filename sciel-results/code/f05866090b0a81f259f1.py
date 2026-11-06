import random
import time
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 110  # 40s safety margin for 150s limit
    
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6."""
        nu = adj[u]
        nv = adj[v]
        
        # Iterate a in N(u)
        nu_iter = nu
        while nu_iter:
            a_bit = nu_iter & (-nu_iter)
            a = a_bit.bit_length() - 1
            nu_iter ^= a_bit
            
            # Iterate b in N(v) \ {a}
            nv_iter = nv & ~(1 << a)
            while nv_iter:
                b_bit = nv_iter & (-nv_iter)
                b = b_bit.bit_length() - 1
                nv_iter ^= b_bit
                
                mask = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                Sa = adj[a] & ~mask
                Sb = adj[b] & ~mask
                
                if Sa == 0 or Sb == 0:
                    continue
                
                # Check if there is an edge (x, y) with x in Sa, y in Sb, x != y
                sa_iter = Sa
                while sa_iter:
                    x_bit = sa_iter & (-sa_iter)
                    x = x_bit.bit_length() - 1
                    sa_iter ^= x_bit
                    
                    # If x is connected to any node in Sb (other than x), we have a cycle
                    # Note: x is in Sa, so x != u, v, a, b.
                    # If x in Sb, we need a neighbor y in Sb \ {x}.
                    # If x not in Sb, we need a neighbor y in Sb.
                    # adj[x] & Sb & ~(1 << x) covers both cases.
                    if adj[x] & Sb & ~(1 << x):
                        return True
        return False

    # Randomized Greedy
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [0] * N
        edge_set = set()
        
        for u, v in all_edges:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        current_count = len(edge_set)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            
            # If we found a very high count, we can break early if we are close to deadline
            # but let's keep searching to see if we can go higher.
            # 105 is the known lower bound. If we find 106, it's a breakthrough.
            if best_count > 105:
                # Keep searching for a bit longer to confirm or improve
                pass

    return [[u, v] for u, v in best_edges]
