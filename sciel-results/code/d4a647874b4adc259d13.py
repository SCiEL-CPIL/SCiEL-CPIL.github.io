import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 90  # Safer margin
    
    # Precompute all edges
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_add(masks, u, v):
        mu = masks[u]
        mv = masks[v]
        
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if adding u-v creates a K3,3 subgraph
            # K3,3 exists if there are 3 common neighbors between two sets of 3.
            # Specifically, if we add (u,v), we need to ensure no pair of vertices has >2 common neighbors.
            # Actually, the condition "no three vertices have three common neighbours" is equivalent to
            # no K3,3.
            # Adding (u,v) affects common neighbors of (u, w) and (v, w) for all w.
            # Also affects (u, v) itself? No, u and v are not neighbors before.
            # Common neighbors of u and v: (mu & mv). If this was >2, it was already invalid.
            # New common neighbors of u and w: (mu | (1<<v)) & masks[w].
            # New common neighbors of v and w: (mv | (1<<u)) & masks[w].
            
            if ((mu | (1 << v)) & masks[w]).bit_count() > 2:
                return False
            if ((mv | (1 << u)) & masks[w]).bit_count() > 2:
                return False
        return True

    best_edges_count = 0
    best_masks = [0] * n
    stagnation_count = 0
    max_stagnation = 50 # If no improvement in 50 restarts, break
    
    while time.time() < deadline:
        masks = [0] * n
        random.shuffle(edges)
        current_count = 0
        
        for u, v in edges:
            if is_valid_add(masks, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_count += 1
        
        if current_count > best_edges_count:
            best_edges_count = current_count
            best_masks = masks[:]
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count > max_stagnation:
                break

    # Convert best_masks to list of edges
    result = []
    for u in range(n):
        m = best_masks[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                result.append([u, v])
            m ^= lsb
            
    # Final validation and cleanup
    # Ensure no duplicates and valid vertices
    # The mask construction ensures simple graph.
    return result
