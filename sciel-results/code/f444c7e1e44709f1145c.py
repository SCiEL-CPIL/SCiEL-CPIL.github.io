"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""

def construct():
    import itertools
    import random

    n = 9
    # Start with complete 5-uniform hypergraph on 9 vertices
    all_edges = list(itertools.combinations(range(n), 5))
    
    def is_not_2_colorable(n, edges):
        """
        Returns True if the hypergraph is NOT 2-colorable.
        Uses bitmask enumeration of all 2^n colorings.
        """
        if not edges:
            return False
            
        # Precompute edge masks
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        full_mask = (1 << n) - 1
        
        # Iterate through all possible subsets A (representing vertices colored 1)
        # B = V \ A (vertices colored 0)
        # A coloring is valid (2-coloring) if no edge is monochromatic.
        # i.e., for every edge E, E is not subset of A AND E is not subset of B.
        # If we find such a coloring, the hypergraph IS 2-colorable.
        
        for A_mask in range(1 << n):
            B_mask = full_mask ^ A_mask
            is_valid_coloring = True
            for em in edge_masks:
                # Check if E is subset of A
                if (em & A_mask) == em:
                    is_valid_coloring = False
                    break
                # Check if E is subset of B
                if (em & B_mask) == em:
                    is_valid_coloring = False
                    break
            if is_valid_coloring:
                return False # Hypergraph is 2-colorable
        
        return True # Hypergraph is NOT 2-colorable

    current_edges = all_edges[:]
    
    # Greedy removal with randomization
    # Try multiple passes or random shuffles to escape local minima
    for _ in range(10): # 10 attempts
        random.shuffle(current_edges)
        i = 0
        while i < len(current_edges):
            # Try to remove current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            if is_not_2_colorable(n, test_edges):
                # Still not 2-colorable, so we can remove it
                current_edges = test_edges
                # Do not increment i, check the next one in the new list
            else:
                i += 1
                
    # Final verification
    if not is_not_2_colorable(n, current_edges):
        # Fallback to complete hypergraph if something went wrong (shouldn't happen)
        return {'n': n, 'edges': [list(e) for e in all_edges]}
        
    return {'n': n, 'edges': [list(e) for e in current_edges]}
