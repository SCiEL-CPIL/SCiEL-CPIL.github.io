"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_triangle_free(n, adj_mask):
    """Check if graph is triangle-free using adjacency bitmasks."""
    for u in range(n):
        # For each neighbor v of u, check if they share a common neighbor
        # Actually, just check if adj[u] & adj[v] is non-empty for v in adj[u]
        # But we must ensure v > u to avoid double checking and self-loops logic
        # Let's iterate neighbors
        neighbors_u = adj_mask[u]
        # Iterate through set bits of neighbors_u
        v = 0
        while neighbors_u:
            if neighbors_u & 1:
                # v is a neighbor
                # Check if u and v have common neighbors
                # Common neighbors are in adj_mask[u] & adj_mask[v]
                # We need to ensure no triangle u-v-w
                # If (adj_mask[u] & adj_mask[v]) != 0, there is a common neighbor.
                # However, we must be careful: the common neighbor w must form a triangle.
                # In a simple graph, if u~v and w is common neighbor, then u~w and v~w.
                # So u-v-w is a triangle.
                if adj_mask[u] & adj_mask[v]:
                    return False
            neighbors_u >>= 1
            v += 1
    return True

def _max_independent_set_size(n, adj_mask):
    """
    Exact branch and bound for maximum independent set size.
    Uses bitmasks for speed.
    """
    best = 0
    # Pre-compute candidate sets? No, dynamic.
    
    def branch(candidates_mask, size):
        nonlocal best
        if size + bin(candidates_mask).count('1') <= best:
            return
        if not candidates_mask:
            if size > best:
                best = size
            return
        
        # Select a vertex to branch on
        # Heuristic: pick vertex with minimum degree in the induced subgraph?
        # Or maximum? For independent set, we want to remove many vertices.
        # If we pick v, we remove v and its neighbors from candidates if we include v.
        # So picking a vertex with high degree in candidates is good for pruning the "include" branch.
        
        # Find vertex with max degree in candidates
        max_deg = -1
        chosen = -1
        cm = candidates_mask
        v_idx = 0
        while cm:
            if cm & 1:
                # Count neighbors of v_idx in candidates_mask
                deg = bin(adj_mask[v_idx] & candidates_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v_idx
            cm >>= 1
            v_idx += 1
            
        if chosen == -1:
             return

        # Branch 1: Include chosen
        # New candidates: candidates without chosen and its neighbors
        new_cand = candidates_mask & ~(adj_mask[chosen] | (1 << chosen))
        branch(new_cand, size + 1)
        
        # Branch 2: Exclude chosen
        new_cand = candidates_mask & ~(1 << chosen)
        branch(new_cand, size)

    branch((1 << n) - 1, 0)
    return best

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Seed solution
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_n = n_seed
    best_edges = edges_seed
    
    # Helper to convert adj_mask to edges list
    def mask_to_edges(n, mask_list):
        edges = []
        for i in range(n):
            m = mask_list[i]
            j = i + 1
            while m:
                if m & 1:
                    edges.append([i, j])
                m >>= 1
                j += 1
        return edges

    # Search for larger n using Circulant Graphs
    # We try n from 40 down to 27
    for target_n in range(40, 26, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        attempts = 0
        max_attempts = 2000  # Limit attempts per n to avoid timeout
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 5:
                break
                
            n = target_n
            half_n = n // 2
            # Choose a random set S of offsets
            # Size of S should be such that graph is dense enough to reduce alpha
            # but sparse enough to avoid triangles.
            # Typical degree for R(3,10) graphs is around n/2?
            # Let's try sizes between 3 and half_n-1
            min_k = 3
            max_k = half_n - 1
            if min_k > max_k:
                continue
                
            k = random.randint(min_k, max_k)
            # Randomly select k offsets from 1..half_n
            # Ensure symmetry: if i in S, n-i is effectively handled by undirected nature
            # In circulant graph on Z_n, edge (u,v) exists if (v-u)%n in S or (u-v)%n in S.
            # If we pick S from 1..floor(n/2), we define the undirected edges uniquely.
            
            offsets = random.sample(range(1, half_n + 1), k)
            
            # Build adj_mask
            adj_mask = [0] * n
            for u in range(n):
                for off in offsets:
                    v = (u + off) % n
                    adj_mask[u] |= (1 << v)
                    # Since it's undirected and we only add positive offsets up to n/2,
                    # the reverse direction is covered when we process v with offset -off (which is n-off).
                    # Wait, if we only add (u, u+off), we need to ensure (u+off, u) is also set.
                    # adj_mask[u] has bit v.
                    # adj_mask[v] should have bit u.
                    # (v - u) % n = off.
                    # (u - v) % n = n - off.
                    # If n-off is not in offsets, we haven't added the reverse edge explicitly in this loop for v.
                    # So we must add it.
                    adj_mask[v] |= (1 << u)
            
            # Check triangle free
            if not _is_triangle_free(n, adj_mask):
                continue
                
            # Check independent set
            max_ind = _max_independent_set_size(n, adj_mask)
            if max_ind < 10:
                # Valid!
                if n > best_n:
                    best_n = n
                    best_edges = mask_to_edges(n, adj_mask)
                found = True
                break
        
        if found:
            # If we found a valid graph for this n, we can break if n is high enough?
            # The task asks to maximize n. If we found n=40, we are done.
            # If we found n=39, we should still try n=40? 
            # The loop goes 40, 39... so if we found 40, we break outer loop?
            # Yes, if we find a valid graph for the current target_n (which starts at 40), 
            # and it's the first one, it's the max possible.
            # If we fail 40, we try 39. If we find 39, we should stop? 
            # No, we want the MAXIMUM n. So if we find 39, we still need to check if 40 exists?
            # But we already tried 40 and failed (or timed out). 
            # So if we find 39, we can return it as it's better than 26.
            # However, if we found 40, we are done.
            # The loop structure `for target_n in range(40, 26, -1)` handles this.
            # If we find a valid graph for target_n, we update best.
            # We should break the outer loop if target_n == 40 and found.
            # If target_n < 40, we might still have time to try higher? No, we went down.
            # So if we find 39, we return 39. We don't need to search 38.
            # So we can break the outer loop once we find ANY valid graph for the current target_n?
            # Yes, because we are iterating from high to low. The first valid one found is the max n.
            break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
