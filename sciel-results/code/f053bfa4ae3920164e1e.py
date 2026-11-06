import time
import random
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    # adj[u] is a bitmask of neighbors of u
    # Initialize with K_{6,6,6}
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Precompute triangle masks for current adj
    # tri_mask[u][v] is bitmask of w such that u,v,w form a triangle
    # We will update this when edges flip.
    
    def is_k4_free(adj):
        # Check if any K4 exists
        # A K4 is u,v,w,z such that all 6 edges exist.
        # Iterate pairs (u,v) with edge.
        # Common neighbors w in N(u) & N(v).
        # For each w, check if N(w) & N(u) & N(v) has an edge?
        # Simpler: For each u,v with edge, let C = N(u) & N(v).
        # If C contains an edge (w,z), then u,v,w,z is a K4.
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Check if common contains an edge
                    # Iterate w in common
                    w_mask = common
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        # Check if w has a neighbor in common \ {w}
                        if adj[w] & (common ^ (1 << w)):
                            return False
        return True

    def calc_score(adj):
        # Find max size of triangle-free induced subgraph
        # Iterate subsets from largest to smallest?
        # Or use the hypergraph independent set approach.
        # For N=18, we can just check all subsets.
        # But we want the MAX size.
        # We can iterate k from 18 down to 1.
        # Check if there is a subset of size k that is triangle-free.
        
        # Precompute tri_mask for this adj?
        # tri_mask[u][v] = adj[u] & adj[v] (if edge u-v exists)
        # Actually, if u-v is not an edge, tri_mask is 0.
        
        tri_masks = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    tri_masks[u][v] = adj[u] & adj[v]
                    tri_masks[v][u] = tri_masks[u][v]
        
        # Check subsets of size k
        # To be efficient, we can use a recursive search or just iterate combinations.
        # Combinations of 18 choose 12 is 18564.
        # Combinations of 18 choose 11 is 31824.
        # This is small enough.
        
        from itertools import combinations
        
        # We want to return the largest k such that SOME subset of size k is triangle-free.
        # We can check k=18, 17, ...
        # But we can start from a known good upper bound.
        # The seed is 12. Let's check down from 12.
        
        for k in range(18, 0, -1):
            found = False
            for subset in combinations(range(n), k):
                s_mask = 0
                for v in subset:
                    s_mask |= 1 << v
                
                # Check if s_mask is triangle-free
                is_tf = True
                # For each pair u,v in subset with edge
                # Check if s_mask & tri_masks[u][v] != 0
                # We can iterate u in subset
                sub_list = list(subset)
                for i in range(len(sub_list)):
                    u = sub_list[i]
                    for j in range(i+1, len(sub_list)):
                        v = sub_list[j]
                        if (adj[u] >> v) & 1:
                            if s_mask & tri_masks[u][v]:
                                is_tf = False
                                break
                    if not is_tf:
                        break
                if is_tf:
                    found = True
                    break
            if found:
                return k
        return 0

    best_adj = adj[:]
    best_score = calc_score(adj)
    
    # Simulated Annealing
    current_adj = adj[:]
    current_score = best_score
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        # Pick random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip
        current_adj[u] ^= 1 << v
        current_adj[v] ^= 1 << u
        
        # Check K4-free
        if not is_k4_free(current_adj):
            # Revert
            current_adj[u] ^= 1 << v
            current_adj[v] ^= 1 << u
            continue
            
        # Calculate score
        new_score = calc_score(current_adj)
        
        # Accept if better or with probability
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            # Revert
            current_adj[u] ^= 1 << v
            current_adj[v] ^= 1 << u
            
        T *= cooling_rate

    return get_edges(best_adj)

if __name__ == "__main__":
    construct()
