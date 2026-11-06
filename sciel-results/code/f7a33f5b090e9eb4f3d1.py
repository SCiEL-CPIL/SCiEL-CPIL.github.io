"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 40
    
    # Use integer bitmasks for fast bitset operations
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def is_c6_free(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 is u-a-b-c-d-v-u.
        We need to check if there exist distinct a,b,c,d such that:
        a in N(u), d in N(v), b in N(a), c in N(d), b-c is an edge.
        And a,b,c,d are distinct and not u,v.
        
        Efficient check:
        1. Get neighbors of u (excluding v) and neighbors of v (excluding u).
        2. For each a in N(u) and d in N(v) with a != d:
           We need a path a-b-c-d of length 3.
           This is equivalent to checking if there is an edge (b,c) 
           such that b in N(a) \ {u,v,d} and c in N(d) \ {u,v,a,b}.
           
           Actually, simpler: 
           Let S_a = N(a) \ {u, v, d}
           Let S_d = N(d) \ {u, v, a}
           If there exists b in S_a and c in S_d such that adj[b] & (1<<c) is true, 
           then we have a C6.
           
           Even faster:
           Compute the set of vertices reachable from 'a' in 2 steps within the graph 
           excluding u,v, and check if any of them is adjacent to 'd' (excluding a).
           
           Let's stick to the bitset intersection method which is very fast in Python/NumPy if we precompute or use integers.
           Since n=40, we can use Python integers as bitsets.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # Iterate through pairs (a, d)
        # a in nu, d in nv
        a_list = [i for i in range(n) if (nu_mask >> i) & 1]
        d_list = [i for i in range(n) if (nv_mask >> i) & 1]
        
        for a in a_list:
            for d in d_list:
                if a == d:
                    continue
                
                # We need a path a-b-c-d
                # b must be neighbor of a, not u,v,a,d
                # c must be neighbor of d, not u,v,a,d
                # and edge (b,c) must exist
                
                # Candidates for b: neighbors of a excluding u, v, d
                # Note: a is not a neighbor of itself in simple graph, so no need to exclude a from N(a)
                b_candidates = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d)
                
                # Candidates for c: neighbors of d excluding u, v, a
                c_candidates = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # We need to check if there is an edge between any b in b_candidates and any c in c_candidates.
                # This is equivalent to checking if (union of neighbors of all b in b_candidates) intersects c_candidates.
                # Or simpler: for each b in b_candidates, check if adj[b] intersects c_candidates.
                
                # Iterate through b_candidates
                b_mask = b_candidates
                while b_mask:
                    # Get lowest set bit
                    b = (b_mask & -b_mask).bit_length() - 1
                    b_mask ^= (1 << b)
                    
                    # Check if any neighbor of b is in c_candidates
                    if adj[b] & c_candidates:
                        return False
                        
        return True

    def count_edges():
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    # Strategy: Start with a strong seed to avoid getting stuck in sparse local optima.
    # 8 disjoint K5s is a known C6-free structure with 8 * 10 = 80 edges.
    # We will then greedily add all other possible edges in a fixed order (or shuffled) 
    # but using the fast bitset check, we can likely do a full sweep quickly.
    
    best_edges = []
    best_count = 0
    
    # Multiple restarts with different seeds and strategies
    for restart in range(20):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        edges = []
        
        # Initialize with 8 disjoint K5s
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Now try to add all other edges
        # Shuffle the remaining edges to break symmetry
        remaining_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    remaining_edges.append((u, v))
        
        random.seed(42 + restart)
        random.shuffle(remaining_edges)
        
        added_in_pass = 0
        # We might need multiple passes because adding one edge might open up opportunities?
        # No, adding edges only makes it HARDER to add more (more constraints).
        # So a single greedy pass is optimal for a given order.
        # However, different orders yield different results.
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if is_c6_free(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                added_in_pass += 1
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edges)
            
        # Early stop if we hit the theoretical best or close to it
        if best_count >= 105:
            break

    # Final verification step to ensure no C6 (optional but good for safety)
    # The is_c6_free check is exact, so the result should be valid.
    
    return best_edges
# EVOLVE-BLOCK-END
