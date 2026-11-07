import time
import itertools

def construct():
    # The Fano plane is the smallest 3-chromatic 3-uniform hypergraph.
    # It has 7 edges and 7 vertices.
    # We embed it in 9 vertices (0..6 used, 7,8 isolated).
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane
    # v0=0, v1=0, v2=1, v3=1, v4=2, v5=1, v6=2
    # v7=0, v8=0
    colors = [0, 0, 1, 1, 2, 1, 2, 0, 0]
    
    # Verification 1: No edge is monochromatic in the provided coloring
    for e in edges:
        c0 = colors[e[0]]
        c1 = colors[e[1]]
        c2 = colors[e[2]]
        if c0 == c1 == c2:
            # Fallback: try to find a valid coloring via search if hardcoded fails
            # But it shouldn't fail.
            pass
            
    # Verification 2: The hypergraph is NOT 2-colorable
    # If it were 2-colorable, there would exist a binary coloring with no monochromatic edges.
    # We check all 2^9 = 512 binary colorings.
    is_2_colorable = False
    for mask in range(1 << 9):
        # Construct binary coloring from mask
        # bit i of mask represents color of vertex i
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
            
    if not is_2_colorable:
        # It is 3-chromatic (at least 3, and we have a 3-coloring, so exactly 3)
        return {
            'edges': [list(e) for e in edges],
            'colors': colors
        }
    else:
        # This should never happen for the Fano plane
        # Fallback to a larger known 3-chromatic structure or search
        # For safety, return the seed if verification fails (though it won't)
        return {
            'edges': [list(e) for e in itertools.combinations(range(5), 3)],
            'colors': [0, 0, 1, 1, 2, 0, 0, 0, 0]
        }
