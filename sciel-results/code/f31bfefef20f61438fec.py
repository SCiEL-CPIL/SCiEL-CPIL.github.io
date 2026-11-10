"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {}
    for e in all_edges:
        edge_index[e] = True
    
    # Bitmask representation for adjacency
    # adj[i] is a bitmask where bit j is set if edge (i,j) exists
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        u, v are the endpoints of the new edge (not yet in adj).
        """
        # Neighbors of u excluding v
        neigh_u = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        neigh_v = adj[v] & ~(1 << u)
        
        # Iterate over a2, a4 (neighbors of u, distinct)
        # a2 must be connected to some a3 (neighbor of v)
        # a4 must be connected to some a5 (neighbor of v)
        
        # For each pair (a2, a4) from neighbors of u:
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                # a3 must be neighbor of v AND neighbor of a2
                a3_candidates = neigh_v & adj[a2_idx]
                # a5 must be neighbor of v AND neighbor of a4
                a5_candidates = neigh_v & adj[a4_idx]
                
                # a3 != a2, a3 != a4
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                # a5 != a2, a5 != a3, a5 != a4
                # We'll handle a5 != a3 in the loop
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        # a6: common neighbor of a2 and a4, not in {u,v,a2,a3,a4,a5}
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        # a7: common neighbor of a3 and a5, not in {u,v,a2,a3,a4,a5,a6}
                        # We need a6 and a7 to be connected
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            # Check if any a7 in a7_cand is connected to a6
                            if adj[a6_idx] & a7_cand:
                                return True
        return False
    
    def greedy_add(adj, edge_order):
        """Add edges in given order, skipping those that create Q3."""
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added
    
    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result
    
    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2
    
    random.seed(12345)
    best_adj = [0] * n
    best_count = -1
    
    # Phase 1: Multiple greedy random restarts
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Phase 2: Local search - try to add remaining edges to best
    if time.time() < deadline - 20:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        trial = best_adj[:]
        greedy_add(trial, remaining)
        cnt = count_edges(trial)
        if cnt > best_count:
            best_count = cnt
            best_adj = trial
    
    # Phase 3: More random restarts with different seeds
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Try to improve by adding remaining
        if time.time() > deadline - 10:
            break
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (adj[u] & (1 << v))]
        if remaining:
            random.shuffle(remaining)
            greedy_add(adj, remaining)
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]
    
    # Phase 4: Final attempt to add edges to best
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)
    
    return edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
