import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    m = len(all_edges)  # 120 edges

    # Q3 structure
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Precompute neighbors for Q3 vertices
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    def q3_check_fast(adj_bits, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj_bits creates a Q3.
        Assumes (u,v) is not currently in adj_bits.
        """
        # Map u -> 0, v -> 1 in Q3
        # Neighbors of 0 in Q3 (excluding 1): 2, 4
        # Neighbors of 1 in Q3 (excluding 0): 3, 5
        
        # Candidates for Q3 vertices 2,4 must be neighbors of u in G (excluding v)
        # Candidates for Q3 vertices 3,5 must be neighbors of v in G (excluding u)
        
        mask_u = adj_bits[u] & ~(1 << v)
        mask_v = adj_bits[v] & ~(1 << u)
        
        # Extract lists of neighbors
        nbs_u = [i for i in range(n) if mask_u & (1 << i)]
        nbs_v = [i for i in range(n) if mask_v & (1 << i)]
        
        # We need to assign a,b to 2,4 and c,d to 3,5
        # a,b from nbs_u, c,d from nbs_v
        # Conditions:
        # (a,c) in G, (b,d) in G
        # There exist e,f (6,7) such that:
        # (a,e), (b,e), (c,f), (d,f), (e,f) in G
        # e,f not in {u,v,a,b,c,d}
        
        # Optimization: Precompute common neighbors for pairs in nbs_u and nbs_v?
        # Or just iterate.
        
        # Iterate over pairs (a,b) from nbs_u
        for i_a in range(len(nbs_u)):
            a = nbs_u[i_a]
            for i_b in range(i_a + 1, len(nbs_u)):
                b = nbs_u[i_b]
                
                # Common neighbors of a and b (candidates for 6)
                # Must be in G
                common_ab = []
                mask_common_ab = adj_bits[a] & adj_bits[b]
                # Exclude u, v, a, b
                mask_common_ab &= ~(1 << u)
                mask_common_ab &= ~(1 << v)
                mask_common_ab &= ~(1 << a)
                mask_common_ab &= ~(1 << b)
                
                # We will check candidates for 6 later.
                
                # Iterate over pairs (c,d) from nbs_v
                for i_c in range(len(nbs_v)):
                    c = nbs_v[i_c]
                    # Check (a,c) edge
                    if not (adj_bits[a] & (1 << c)):
                        continue
                    for i_d in range(i_c + 1, len(nbs_v)):
                        d = nbs_v[i_d]
                        # Check (b,d) edge
                        if not (adj_bits[b] & (1 << d)):
                            continue
                        
                        # Exclude a,b,c,d from common_ab to find candidates for 6
                        mask_6 = mask_common_ab
                        mask_6 &= ~(1 << c)
                        mask_6 &= ~(1 << d)
                        
                        # Candidates for 7 must be common neighbors of c and d
                        mask_cd = adj_bits[c] & adj_bits[d]
                        mask_cd &= ~(1 << u)
                        mask_cd &= ~(1 << v)
                        mask_cd &= ~(1 << a)
                        mask_cd &= ~(1 << b)
                        mask_cd &= ~(1 << c)
                        mask_cd &= ~(1 << d)
                        
                        # We need e in mask_6 and f in mask_cd such that (e,f) is an edge
                        # And e != f
                        # Iterate e in mask_6
                        e_mask = mask_6
                        while e_mask:
                            lsb = e_mask & -e_mask
                            e = lsb.bit_length() - 1
                            e_mask ^= lsb
                            
                            # Check if e is connected to any f in mask_cd
                            # f must be in mask_cd and adjacent to e
                            f_mask = mask_cd & adj_bits[e]
                            if f_mask:
                                # Found a Q3
                                return True
        
        return False

    def count_edges(adj_bits):
        return sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2

    def init_k313():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_count = 0
    best_adj = [0] * n
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        random.seed(trial * 104729 + 7)
        
        if trial % 5 == 0:
            adj = init_k313()
        else:
            adj = [0] * n
            for u, v in all_edges:
                if random.random() < 0.3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Greedy add
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if not q3_check_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local Search
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding missing edges
            for u, v in all_edges:
                if not (adj[u] & (1 << v)):
                    if not q3_check_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
            
            if not improved:
                # Try 1-swap
                current_count = count_edges(adj)
                existing = [(u, v) for u, v in all_edges if adj[u] & (1 << v)]
                random.shuffle(existing)
                
                for e1 in existing:
                    if time.time() > deadline:
                        break
                    u1, v1 = e1
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    
                    for u2, v2 in all_edges:
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        if not (adj[u2] & (1 << v2)):
                            if not q3_check_fast(adj, u2, v2):
                                adj[u2] |= (1 << v2)
                                adj[v2] |= (1 << u2)
                                new_count = count_edges(adj)
                                if new_count > current_count:
                                    improved = True
                                    break
                                else:
                                    adj[u2] &= ~(1 << v2)
                                    adj[v2] &= ~(1 << u2)
                    
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
