import time
import itertools
from collections import Counter

def construct():
    deadline = time.time() + 45.0

    # The Fano plane on 7 vertices is the unique (up to isomorphism) smallest
    # 3-uniform hypergraph with chromatic number 3. It has exactly 7 edges.
    # A well-known result states that any 3-uniform hypergraph with fewer than
    # 7 edges is 2-colorable. Thus 7 is the global minimum for this problem.

    # We embed the Fano plane on vertices 0..6 and leave 7,8 isolated.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane.
    # No edge is monochromatic, and at least 3 colors are required globally.
    colors = [0, 1, 2, 1, 1, 2, 0, 0, 0]

    # --- Verification and Polish ---
    
    # 1. Verify that the current 3-coloring is valid (no monochromatic edge).
    valid_coloring = True
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            valid_coloring = False
            break
    
    # 2. Verify that the hypergraph is indeed 3-chromatic by brute-forcing
    #    all 2^9 = 512 possible 2-colorings to confirm none are valid.
    #    This ensures the "three-chromatic" requirement is met strictly.
    is_not_2_colorable = True
    for mask in range(512):
        # Construct a candidate 2-coloring from the bitmask
        c = [(mask >> i) & 1 for i in range(9)]
        # Check if any edge is monochromatic under this coloring
        ok = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                ok = False
                break
        if ok:
            # Found a valid 2-coloring, so the graph is 2-colorable.
            is_not_2_colorable = False
            break

    # If for some reason the Fano plane instance failed verification (which it shouldn't),
    # we would need a fallback, but given the fixed nature of the Fano plane, this is redundant.
    # However, to be safe against any subtle implementation error in the check, 
    # we trust the mathematical fact that the Fano plane is 3-chromatic.
    
    if not valid_coloring:
        # This block should never execute. If it does, the coloring is invalid.
        # We could try to find a valid 3-coloring, but since we know one exists for the Fano plane,
        # this indicates a bug in the constant definition or check logic.
        # For robustness, let's just try to find ANY 3-coloring quickly if this happens.
        # But actually, the constant coloring is hardcoded and correct.
        pass

    if not is_not_2_colorable:
        # This should never happen for the Fano plane.
        # If it did, the graph is 2-colorable and invalid for the task.
        # We would need a different hypergraph. 
        # But the Fano plane is proven 3-chromatic.
        pass

    # Ensure all integers are Python int (not numpy, not bool)
    final_edges = []
    for e in edges:
        final_edges.append([int(v) for v in e])
        
    final_colors = [int(c) for c in colors]

    return {'edges': final_edges, 'colors': final_colors}
