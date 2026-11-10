import itertools
import time

def _colorable_fast(n, edges):
    """
    Fast check: is the hypergraph 2-colorable?
    Returns True if 2-colorable (i.e., there exists a coloring with no monochromatic edge).
    Returns False if not 2-colorable.
    
    Uses bitmask backtracking with unit propagation and a strong vertex selection heuristic.
    """
    if not edges:
        return True

    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1

    def propagate(color1, unassigned):
        """
        Apply unit propagation. 
        Returns (color1, unassigned, conflict_flag).
        conflict_flag True means a monochromatic edge was found.
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                unassigned_in_edge = mask & unassigned
                assigned_in_edge = mask & ~unassigned

                # Case 1: All 5 assigned
                if unassigned_in_edge == 0:
                    c1_count = bin(assigned_in_edge & color1).count('1')
                    if c1_count == 0 or c1_count == 5:
                        return None, None, True
                    continue

                # Case 2: Exactly 1 unassigned
                if unassigned_in_edge & (unassigned_in_edge - 1) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    c1_count = bin(other_4 & color1).count('1')
                    if c1_count == 4:
                        # Force unassigned to color 0 (i.e., not in color1)
                        # Just remove from unassigned (it stays out of color1)
                        unassigned ^= unassigned_in_edge
                        changed = True
                    elif c1_count == 0:
                        # Force unassigned to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True

        return color1, unassigned, False

    def get_heuristic_vertex(color1, unassigned):
        """
        Pick the vertex to assign next.
        Heuristic: maximize the number of "critical" edges (edges with exactly 1 unassigned vertex) that contain this vertex.
        If tied, pick the lowest index vertex.
        """
        # Count critical edges per vertex
        counts = [0] * n
        for mask in edge_masks:
            unassigned_in_edge = mask & unassigned
            if unassigned_in_edge & (unassigned_in_edge - 1) == 0:
                # Single unassigned vertex
                v = unassigned_in_edge.bit_length() - 1
                counts[v] += 1

        best_v = -1
        best_count = -1
        for v in range(n):
            if unassigned & (1 << v):
                if counts[v] > best_count:
                    best_count = counts[v]
                    best_v = v

        if best_v == -1:
            # No critical edges, pick lowest unassigned
            return unassigned & -unassigned
        else:
            return 1 << best_v

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False

        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            # Propagation should have caught this, but double check for safety.
            for mask in edge_masks:
                c1_count = bin(mask & color1).count('1')
                if c1_count == 0 or c1_count == 5:
                    return False
            return True

        v_bit = get_heuristic_vertex(color1, unassigned)

        # Try color 1 (add to color1)
        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True

        # Try color 0 (do not add to color1)
        if backtrack(color1, unassigned ^ v_bit):
            return True

        return False

    # Symmetry breaking: Fix vertex 0 to color 0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1

    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return False

    if initial_unassigned == 0:
        for mask in edge_masks:
            c1_count = bin(mask & initial_color1).count('1')
            if c1_count == 0 or c1_count == 5:
                return False
        return False

    if backtrack(initial_color1, initial_unassigned):
        return True

    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # We will try to find a minimal non-2-colorable hypergraph by starting with K_9^5 and pruning.
    # But to be more effective, we can try to restart pruning with different random orders or even different initial subsets.
    # However, the standard approach is to start with the complete hypergraph and remove edges one by one if possible.
    
    best_edges = None
    best_n = 9

    # Iterate over a few random shuffles to find a better local minimum
    for attempt in range(10): # Max 10 attempts, or until time limit
        if time.time() > deadline - 5:
            break

        n = 9
        all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        
        # Shuffle to get different pruning orders
        import random
        random.seed(attempt * 12345 + 67890)
        random.shuffle(all_edges)
        
        current_edges = all_edges[:]
        
        # Pruning loop
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Iterate from end to start to allow safe removal
            for i in range(len(current_edges) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                
                # Try removing current_edges[i]
                new_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if new_edges is NOT 2-colorable
                if not _colorable_fast(n, new_edges):
                    current_edges = new_edges
                    changed = True
            
            # If no changes in a full pass, we have a minimal set for this random order
            # We can break out to try a new random order if we want to explore more
            # But typically one pass is enough to get close to minimal for a given order.
            # To be thorough, we continue looping until no changes in a full pass.
            
        if best_edges is None or len(current_edges) < len(best_edges):
            best_edges = current_edges
            best_n = n
            
            # Early exit if we reach the theoretical lower bound or known best
            if len(best_edges) <= 51:
                break

    if best_edges is None:
        # Fallback: return K_9^5
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    # Final verification
    if _colorable_fast(best_n, best_edges):
        # This should not happen if logic is correct, but as a safeguard
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}

    return {'n': best_n, 'edges': best_edges}
