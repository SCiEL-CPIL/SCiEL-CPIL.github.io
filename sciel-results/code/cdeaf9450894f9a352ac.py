import itertools
import random
import time

def _is_non_2_colorable(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable.
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
    
    # Precompute vertex degrees for heuristic ordering
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            v_bit = m & -m
            v_idx = v_bit.bit_length() - 1
            vertex_degree[v_idx] += 1
            m ^= v_bit

    # We use a recursive backtracking search.
    # State: color1 (bitmask of vertices colored 1), unassigned (bitmask of uncolored vertices)
    # Vertices not in color1 and not in unassigned are color 0.
    
    # Symmetry breaking: Fix vertex 0 to color 0.
    # initial_color1 = 0
    # initial_unassigned = full_mask ^ 1
    
    # To speed up, we can precompute for each edge:
    # - The mask of vertices
    # - List of vertices (for iteration if needed, but bitmask is faster)
    
    def propagate(color1, unassigned):
        """
        Apply unit propagation.
        Returns (new_color1, new_unassigned, conflict_flag).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                # If all 5 are assigned
                if unassigned_in_edge == 0:
                    # Check if monochromatic
                    c1_count = bin(mask & color1).count('1')
                    if c1_count == 5 or c1_count == 0:
                        return None, None, True
                    continue
                
                # If exactly 1 is unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    # Check if other 4 are all color 1
                    if (other_4 & color1) == other_4:
                        # Force this vertex to color 0
                        # It is already not in color1, so just remove from unassigned
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Check if other 4 are all color 0
                    # This means (other_4 & color1) == 0
                    elif (other_4 & color1) == 0:
                        # Force this vertex to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
        
        return color1, unassigned, False

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            # Actually, if we got here without conflict in propagate, 
            # and we checked all-assigned edges in propagate, 
            # we are good? 
            # Wait, propagate checks for conflicts when unassigned_in_edge == 0.
            # If unassigned becomes 0, all edges have unassigned_in_edge == 0.
            # So if no conflict was returned, no edge is monochromatic.
            return True
        
        # Heuristic: Pick vertex with highest degree among unassigned
        # Or pick vertex involved in most "almost satisfied" edges?
        # Let's stick to degree for simplicity and speed, it's effective for n=9.
        
        best_v = -1
        best_deg = -1
        m = unassigned
        while m:
            v_bit = m & -m
            v_idx = v_bit.bit_length() - 1
            if vertex_degree[v_idx] > best_deg:
                best_deg = vertex_degree[v_idx]
                best_v = v_bit
            m ^= v_bit
        
        v_bit = best_v
        
        # Try color 1
        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True
            
        # Try color 0
        if backtrack(color1, unassigned ^ v_bit):
            return True
            
        return False

    # Start search
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1 # Vertex 0 is color 0
    
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True
        
    if initial_unassigned == 0:
        # All assigned by propagation. Check for conflicts (should be handled by propagate)
        # But just in case, if no conflict, it's a valid coloring?
        # If no conflict, it means no edge was monochromatic.
        return False # Valid coloring found (so it IS 2-colorable)
        
    if backtrack(initial_color1, initial_unassigned):
        return False # Valid coloring found
        
    return True # No coloring found (NOT 2-colorable)

def _try_remove_edges(n, edges, deadline):
    """
    Greedily remove edges while maintaining non-2-colorability.
    """
    current_edges = edges[:]
    removed_any = True
    
    while removed_any and (time.time() < deadline):
        removed_any = False
        # Shuffle to break symmetry and find different removals
        random.shuffle(current_edges)
        
        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            if _is_non_2_colorable(n, new_edges):
                current_edges = new_edges
                removed_any = True
    
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_n = None
    best_edges = None
    best_count = float('inf')
    
    # Strategy 1: Start with K9^(5) and prune
    n = 9
    all_edges_k9 = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Run multiple pruning attempts with different random seeds/orders
    for attempt in range(10): # Increase attempts since checker is fast
        if time.time() > deadline:
            break
        
        shuffled = all_edges_k9[:]
        random.seed(attempt) # Reproducibility within run, different per attempt
        random.shuffle(shuffled)
        
        sub_deadline = min(deadline, start_time + 10 + attempt * 5)
        pruned = _try_remove_edges(n, shuffled, sub_deadline)
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_n = n
            best_edges = pruned[:]
            # If we hit the known optimum 51, we can stop early or keep trying to verify
            if best_count <= 51:
                # Try to push lower? Unlikely. 51 is the record.
                # We can break if we have time, but let's finish the current attempt.
                pass

    # If we didn't find anything (should not happen), fallback
    if best_edges is None:
        best_n = 9
        best_edges = all_edges_k9
    
    # Final verification
    if not _is_non_2_colorable(best_n, best_edges):
        # Fallback to K9 if something went wrong
        return {'n': 9, 'edges': all_edges_k9}
    
    return {'n': best_n, 'edges': best_edges}
