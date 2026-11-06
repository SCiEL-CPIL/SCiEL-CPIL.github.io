def is_2_colorable_fast(n, edges):
    if not edges:
        return True
        
    if n > 15:
        # Fallback to a simple recursive DPLL or just return False 
        # to indicate we shouldn't use this n in the main loop.
        # But for safety, let's implement a very simple DPLL.
        # Actually, let's just return False to skip n>15 in the loop.
        return False

    # Pre-compute edge masks
    edge_masks = []
    for e in edges:
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks.append(m)
        
    # Brute force all 2^n colorings
    # Symmetry breaking: Fix vertex 0 to 0.
    # So we iterate masks for vertices 1..n-1.
    # Total 2^(n-1) masks.
    
    for mask in range(1 << (n - 1)):
        # Construct the full mask? 
        # Or just check the condition.
        # Vertex 0 is 0.
        # For an edge mask `em`:
        # Let `em_non0` = em & ~(1 << 0)
        # The colors of vertices in em are determined by `mask` (for v>0) and 0 (for v=0).
        
        # Condition for monochromatic 0:
        # All vertices in em must be 0.
        # This means:
        # 1. If 0 in em, then all other vertices in em must be 0 in mask.
        # 2. If 0 not in em, then all vertices in em must be 0 in mask.
        
        # Condition for monochromatic 1:
        # All vertices in em must be 1.
        # This means:
        # 1. If 0 in em, IMPOSSIBLE (since v0=0).
        # 2. If 0 not in em, then all vertices in em must be 1 in mask.
        
        is_mono = False
        for em in edge_masks:
            if em & 1: # Vertex 0 is in the edge
                # Can only be monochromatic 0.
                # Check if all other bits in em are 0 in mask.
                # mask has bits for 1..n-1.
                # We need to check if (em & ~1) is a subset of ~mask (within n-1 bits).
                # Or simply: (mask & (em & ~1)) == 0
                if (mask & (em & ~1)) != 0:
                    is_mono = False # Not all 0, so not mono 0.
                    # Since v0=0, it can't be mono 1.
                    # So this edge is NOT monochromatic.
                    break
                else:
                    # All other vertices are 0, and v0 is 0.
                    # So this edge IS monochromatic (all 0).
                    is_mono = True
                    break
            else: # Vertex 0 NOT in the edge
                # Can be mono 0 or mono 1.
                # Mono 0: (mask & em) == 0
                # Mono 1: (mask & em) == em
                if (mask & em) == 0:
                    is_mono = True
                    break
                if (mask & em) == em:
                    is_mono = True
                    break
                # Otherwise, not monochromatic.
                
        if not is_mono:
            # Found a valid coloring
            return True
            
    return False
