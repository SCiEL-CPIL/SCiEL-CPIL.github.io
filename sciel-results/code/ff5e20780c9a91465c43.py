import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)

def _is_non_2_colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no valid coloring exists).
    """
    if not edges:
        return False

    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1
    
    # Precompute vertex degrees for heuristic
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            m ^= lsb

    def propagate(color1, unassigned):
        """
        Apply unit propagation. Returns (new_color1, new_unassigned, conflict_flag).
        If conflict, returns (None, None, True).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                # Intersect with assigned vertices
                assigned_in_edge = mask & ~unassigned
                
                # Check if all 5 assigned
                if assigned_in_edge == mask:
                    # If all color 1 or all color 0, conflict
                    if (mask & ~color1 & full_mask) == 0: # All color 1
                        return None, None, True
                    if (mask & color1) == 0: # All color 0
                        return None, None, True
                    continue

                # Check for forcing
                unassigned_in_edge = mask & unassigned
                # If exactly 1 unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    # Are all 4 color 1?
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        # Force to color 0 (do nothing to color1, just remove from unassigned)
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Are all 4 color 0?
                    elif (other_4 & color1) == 0:
                        # Force to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
                        
        return color1, unassigned, False

    def backtrack(color1, unassigned):
        # Check for immediate conflict via propagation
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            # Propagation should have caught these, but double check.
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0: # All color 1
                    return False
                if (mask & color1) == 0: # All color 0
                    return False
            return True # Valid coloring found
            
        # Pick a vertex to assign.
        # Heuristic: Pick vertex with highest degree among unassigned.
        # To avoid O(n) scan every time, we could maintain a priority queue, 
        # but for small n, a simple scan or picking a random high-degree vertex is fine.
        # Let's pick the unassigned vertex with the highest degree.
        best_v_bit = 0
        best_deg = -1
        m = unassigned
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if vertex_degree[idx] > best_deg:
                best_deg = vertex_degree[idx]
                best_v_bit = lsb
            m ^= lsb
        
        v_bit = best_v_bit
        
        # Try color 1
        new_color1 = color1 | v_bit
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned):
            return True
            
        # Try color 0
        new_color1 = color1
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned):
            return True
            
        return False

    # Start search
    # Symmetry breaking: Fix vertex 0 to color 0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1 # Exclude vertex 0
    
    # Propagate initial state
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True
        
    if initial_unassigned == 0:
        # Check validity
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False # Coloring found
        
    if backtrack(initial_color1, initial_unassigned):
        return False # Coloring found
        
    return True # No coloring found

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin

    # Start with K9^(5)
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Optimize: We want to remove edges.
    # We use a greedy approach: try to remove each edge.
    # To make it faster, we can order edges by some heuristic?
    # Or just random shuffle.
    
    # We will loop until no more edges can be removed.
    # Since the checker is fast, we can do many iterations.
    
    # To avoid getting stuck in local minima, we can restart with different shuffles
    # if we have time. But greedy removal is usually good for this size.
    
    removed_any = True
    while removed_any and (time.time() - start_time) < deadline:
        removed_any = False
        # Shuffle edges to try removing in random order
        random.shuffle(current_edges)
        
        # Iterate backwards to allow removal
        for i in range(len(current_edges) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
                
            e = current_edges[i]
            # Try removing e
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if new_edges is non-2-colorable
            if _is_non_2_colorable_fast(n, new_edges):
                # It is still non-2-colorable, so we can remove e
                current_edges = new_edges
                removed_any = True
            # Else, keep e
            
        # If we removed any, loop again to see if more can be removed
        # (Removing one edge might make another redundant)
        
    # Final verification
    # Ensure the result is valid
    if not _is_non_2_colorable_fast(n, current_edges):
        # Fallback to full K9 if something went wrong (shouldn't happen)
        return {'n': n, 'edges': all_edges}
        
    return {'n': n, 'edges': current_edges}
