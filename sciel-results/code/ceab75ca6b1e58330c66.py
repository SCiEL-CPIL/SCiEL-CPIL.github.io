import time
import random
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 115.0  # 5 seconds margin
    
    n = 16
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def get_neighbors(i):
        return adj[i]
        
    def is_connected(u, v):
        return (adj[u] >> v) & 1
        
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def would_create_q3(u, v):
        # Check if adding edge (u, v) creates a Q3
        # 1. Find common neighbors of u and v
        common = adj[u] & adj[v]
        common_list = [i for i in range(n) if (common >> i) & 1]
        
        if len(common_list) < 2:
            return False # No 4-cycle possible
        
        # 2. For each pair (a, b) in common_list, we have a 4-cycle u-a-v-b-u
        # We need to check if this 4-cycle can be extended to a Q3.
        # The extension requires 4 more vertices c, d, e, f such that:
        # (u,c), (a,d), (v,e), (b,f) are edges
        # (c,d), (d,e), (e,f), (f,c) are edges
        
        # To be fast, we can pre-compute potential "wings" for each 4-cycle?
        # Or just check directly.
        
        for a in common_list:
            for b in common_list:
                if a == b:
                    continue
                
                # Cycle is u-a-v-b-u
                # We need to find c, d, e, f
                # c is neighbor of u
                # d is neighbor of a
                # e is neighbor of v
                # f is neighbor of b
                # And c-d-e-f-c is a cycle.
                
                # Let's iterate over possible c (neighbor of u)
                # c cannot be a, b, v (already in cycle)
                # c must be connected to d, which is connected to a and e...
                
                # This is getting expensive to do naively for every pair.
                # Let's optimize:
                # For a fixed 4-cycle (u, a, v, b), the required structure is fixed.
                # We can check if the graph contains a specific pattern.
                
                # Pattern vertices: u, a, v, b are fixed.
                # We need to find a permutation (c, d, e, f) from the remaining 12 vertices.
                # Constraints:
                # 1. (u, c) in E
                # 2. (a, d) in E
                # 3. (v, e) in E
                # 4. (b, f) in E
                # 5. (c, d) in E
                # 6. (d, e) in E
                # 7. (e, f) in E
                # 8. (f, c) in E
                
                # We can iterate over c in N(u) \ {a, b, v}
                # Then d in N(a) \ {u, v, c} and N(c)
                # Then e in N(v) \ {u, b, d} and N(d)
                # Then f in N(b) \ {u, a, e} and N(e) and N(c)
                
                n_u = adj[u]
                n_a = adj[a]
                n_v = adj[v]
                n_b = adj[b]
                
                # Exclude u, v, a, b from candidates
                mask_exclude = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                
                # Candidates for c: N(u) & ~mask_exclude
                c_candidates = n_u & ~mask_exclude
                while c_candidates:
                    c_bit = c_candidates & -c_candidates
                    c = c_bit.bit_length() - 1
                    c_candidates ^= c_bit
                    
                    # Candidates for d: N(a) & N(c) & ~mask_exclude & ~(1<<c)
                    d_candidates = n_a & n_u & ~mask_exclude & ~c_bit
                    # Wait, d must be neighbor of a and c.
                    # d_candidates = n_a & adj[c] & ~mask_exclude & ~c_bit
                    d_candidates = n_a & adj[c] & ~mask_exclude & ~c_bit
                    while d_candidates:
                        d_bit = d_candidates & -d_candidates
                        d = d_bit.bit_length() - 1
                        d_candidates ^= d_bit
                        
                        # Candidates for e: N(v) & N(d) & ~mask_exclude & ~(1<<c) & ~(1<<d)
                        e_candidates = n_v & adj[d] & ~mask_exclude & ~c_bit & ~d_bit
                        while e_candidates:
                            e_bit = e_candidates & -e_candidates
                            e = e_bit.bit_length() - 1
                            e_candidates ^= e_bit
                            
                            # Candidates for f: N(b) & N(e) & N(c) & ~mask_exclude & ~(1<<c) & ~(1<<d) & ~(1<<e)
                            f_candidates = n_b & adj[e] & adj[c] & ~mask_exclude & ~c_bit & ~d_bit & ~e_bit
                            if f_candidates:
                                # If any f exists, we found a Q3
                                return True
                                
        return False

    # Greedy Construction
    best_edges = 0
    best_adj = [0] * n
    
    # Try multiple random orders
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Bias towards bipartite edges first?
    # Let's just shuffle and try.
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        adj = [0] * n
        edge_count = 0
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
                
            if is_connected(u, v):
                continue
                
            if not would_create_q3(u, v):
                add_edge(u, v)
                edge_count += 1
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

