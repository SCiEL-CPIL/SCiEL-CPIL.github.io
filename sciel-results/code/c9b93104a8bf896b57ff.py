import itertools
import random
import time

def _is_non_2_colorable(n, edges):
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
        
    # Remove duplicate masks? The problem says distinct edges, but let's be safe.
    # Actually, distinct edges are guaranteed by input.
    
    full_mask = (1 << n) - 1
    
    # Precompute vertex degrees or incidence for heuristics?
    # For n <= 26, a simple heuristic is enough.
    
    def propagate(color1, unassigned):
        """
        Apply unit propagation. 
        color1: bitmask of vertices colored 1.
        unassigned: bitmask of vertices not yet colored.
        Returns (new_color1, new_unassigned, conflict_flag).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                # Count assigned vertices in this edge
                assigned_in_edge = mask & ~unassigned
                # Count vertices colored 1 in this edge
                c1_in_edge = assigned_in_edge & color1
                # Count vertices colored 0 in this edge
                # Note: ~color1 needs to be masked to n bits
                c0_in_edge = assigned_in_edge & (~color1) & full_mask
                
                # If all 5 are assigned
                if assigned_in_edge == mask:
                    if c1_in_edge == mask: # All 1
                        return None, None, True
                    if c0_in_edge == mask: # All 0
                        return None, None, True
                    # Else mixed, ok
                
                # If 4 assigned, 1 unassigned
                unassigned_in_edge = mask & unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    # Exactly 1 unassigned
                    # The other 4 are assigned
                    other_4 = mask ^ unassigned_in_edge
                    # Check if other 4 are all 1
                    if (other_4 & color1) == other_4:
                        # Force unassigned to 0
                        # It is already not in color1. Just remove from unassigned.
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Check if other 4 are all 0
                    elif (other_4 & color1) == 0:
                        # Force unassigned to 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True
                        
        return color1, unassigned, False

    def backtrack(color1, unassigned):
        # Apply propagation
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        
        if unassigned == 0:
            # All assigned. Check for monochromatic edges.
            # Propagation should have caught immediate conflicts, but let's verify.
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0: # All 1
                    return False
                if (mask & color1) == 0: # All 0
                    return False
            return True # Valid coloring found
            
        # Heuristic: Pick the unassigned vertex that appears in the most edges 
        # with the fewest unassigned vertices (most constrained).
        # Or simply pick the lowest bit for speed? 
        # Let's try a simple degree heuristic: count how many edges are "almost satisfied" (4 assigned) involving this vertex.
        
        # For speed, let's just pick the lowest bit. 
        # n is small, and propagation is strong.
        v_bit = unassigned & -unassigned
        
        # Try color 1
        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True
            
        # Try color 0
        if backtrack(color1, unassigned ^ v_bit):
            return True
            
        return False

    # Symmetry breaking: Fix vertex 0 to color 0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1
    
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
        return False
        
    if backtrack(initial_color1, initial_unassigned):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    n = 9
    # Generate all possible edges
    all_edges_list = [list(e) for e in itertools.combinations(range(n), 5)]
    # Map edge tuple to index for fast lookup
    edge_to_idx = {tuple(e): i for i, e in enumerate(all_edges_list)}
    
    # Start with K9^(5)
    current_edges = all_edges_list[:]
    
    # Phase 1: Greedy Removal
    # Shuffle to randomize
    random.shuffle(current_edges)
    removed_any = True
    while removed_any and (time.time() - start_time) < deadline:
        removed_any = False
        for i in range(len(current_edges) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            if _is_non_2_colorable(n, new_edges):
                current_edges = new_edges
                removed_any = True
                
    # If we already have a very small number, return
    if len(current_edges) <= 55:
         return {'n': n, 'edges': current_edges}

    # Phase 2: Local Search with Swaps
    # We want to minimize the number of edges.
    # If we are stuck in a local minimum (no edge can be removed), we try to swap edges to find a different local minimum.
    
    best_edges = current_edges[:]
    best_size = len(current_edges)
    
    # Precompute list of all edge tuples for random sampling
    all_edges_tuples = [tuple(e) for e in all_edges_list]
    
    while (time.time() - start_time) < deadline:
        # Try to remove edges
        progress = False
        # Try a few random removals
        for _ in range(10):
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            e = current_edges[idx]
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            if _is_non_2_colorable(n, new_edges):
                current_edges = new_edges
                progress = True
                if len(current_edges) < best_size:
                    best_size = len(current_edges)
                    best_edges = current_edges[:]
                
        if not progress:
            # Stuck. Try a swap.
            if len(current_edges) < 2:
                break
            # Pick random edge to remove
            idx_out = random.randint(0, len(current_edges) - 1)
            e_out = current_edges[idx_out]
            
            # Pick random edge to add (must be distinct)
            # To avoid adding an edge that is already in the set, we can pick from all_edges_tuples
            # and check if it's in current_edges.
            # For efficiency, maintain a set of current edges?
            # n=9, 126 edges. Set lookup is fast.
            current_edge_set = set(tuple(e) for e in current_edges)
            
            # Try to find a valid swap
            # We want to swap e_out with e_in.
            # New set: E' = E \ {e_out} U {e_in}
            # We need E' to be non-2-colorable.
            
            # Randomly sample e_in
            swapped = False
            for _ in range(20):
                e_in = random.choice(all_edges_tuples)
                if e_in in current_edge_set:
                    continue
                
                # Construct new edges list
                # Remove e_out, add e_in
                new_edges = current_edges[:]
                # Remove e_out (by value, but we know idx)
                # Since we picked idx_out, we can just replace
                new_edges[idx_out] = list(e_in)
                
                # Check if non-2-colorable
                if _is_non_2_colorable(n, new_edges):
                    current_edges = new_edges
                    current_edge_set = set(tuple(e) for e in current_edges)
                    swapped = True
                    break
                    
            if not swapped:
                # If we can't find a swap, we might be in a very deep local minimum.
                # Try to restart with a different random seed or break?
                # Let's just break to avoid wasting time if stuck for a while.
                # Or try a few more random removals on a copy?
                pass
                
    # Final check and return best found
    # Ensure best_edges is valid
    if not _is_non_2_colorable(n, best_edges):
         # Fallback to current_edges if it's valid
         if _is_non_2_colorable(n, current_edges):
             return {'n': n, 'edges': current_edges}
         # Otherwise, return the initial K9 (always valid)
         return {'n': n, 'edges': all_edges_list}
         
    return {'n': n, 'edges': best_edges}
