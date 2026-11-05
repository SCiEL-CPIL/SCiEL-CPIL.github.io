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

    # Precompute common neighbors for all pairs
    # common[i][j] is a bitmask of vertices k such that (i,k) and (j,k) are edges in the FULL graph K16?
    # No, common neighbors depend on the current graph state.
    # However, the set of *potential* common neighbors is fixed (all other vertices).
    # We need to check against the *current* adjacency.
    
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

    # Helper to iterate bits
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def q3_check_fast(adj_bits, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj_bits creates a Q3.
        Assumes (u,v) is not currently in adj_bits.
        """
        # Map u -> 0, v -> 1 in Q3
        # a, b correspond to Q3 vertices 2, 4 (neighbors of u)
        # c, d correspond to Q3 vertices 3, 5 (neighbors of v)
        
        mask_u = adj_bits[u] & ~(1 << v)
        mask_v = adj_bits[v] & ~(1 << u)
        
        # Get lists of neighbors for iteration
        nbs_u = list(iter_bits(mask_u))
        nbs_v = list(iter_bits(mask_v))
        
        # If we don't have enough neighbors, we can't form Q3
        if len(nbs_u) < 2 or len(nbs_v) < 2:
            return False

        # Iterate over pairs (a,b) from nbs_u
        for i_a in range(len(nbs_u)):
            a = nbs_u[i_a]
            for i_b in range(i_a + 1, len(nbs_u)):
                b = nbs_u[i_b]
                
                # Common neighbors of a and b (candidates for 6)
                # Must be in G
                mask_common_ab = adj_bits[a] & adj_bits[b]
                # Exclude u, v, a, b
                mask_common_ab &= ~(1 << u)
                mask_common_ab &= ~(1 << v)
                mask_common_ab &= ~(1 << a)
                mask_common_ab &= ~(1 << b)
                
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
    
    # Increase trials due to faster check?
    # The check is still O(n^4) in worst case but with small constants.
    # Let's keep 100 trials but optimize the inner loops.
    
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
        
        # Local Search: Just try to add more edges in random order until no more can be added
        # The previous 1-swap was ineffective for maximization.
        # A simple "stochastic greedy" restart or repeated passes might help.
        
        # Pass 1: Random order
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = False
            edges_to_try = all_edges[:]
            random.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if not (adj[u] & (1 << v)):
                    if not q3_check_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
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
