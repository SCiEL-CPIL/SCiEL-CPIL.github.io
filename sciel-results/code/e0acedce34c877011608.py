"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return a 3-uniform hypergraph on 9 vertices with 7 edges (the Fano plane
    embedded on vertices 0..6) that is 3-chromatic (not 2-colorable) and
    admits a valid 3-coloring where every edge is non-monochromatic.
    
    The Fano plane is the unique projective plane of order 2, with 7 points
    and 7 lines. It is well-known that the Fano plane is not 2-colorable
    (i.e., it has chromatic number 3 for 3-uniform hypergraphs in the sense
    that no 2-coloring avoids monochromatic edges), but it IS 3-colorable.
    
    We use vertices 0-6 for the Fano plane and vertices 7-8 are isolated.
    The coloring assigns colors to make every Fano edge non-monochromatic
    while ensuring no 2-coloring exists.
    
    Fano plane edges (on vertices 0..6):
    Using the standard representation:
    {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    
    A valid 3-coloring:
    0->0, 1->1, 2->2, 3->1, 4->2, 5->0, 6->1
    Let's verify each edge:
    {0,1,2}: colors 0,1,2 - non-monochromatic ✓
    {0,3,4}: colors 0,1,2 - non-monochromatic ✓
    {0,5,6}: colors 0,0,1 - non-monochromatic ✓
    {1,3,5}: colors 1,1,0 - non-monochromatic ✓
    {1,4,6}: colors 1,2,1 - non-monochromatic ✓
    {2,3,6}: colors 2,1,1 - non-monochromatic ✓
    {2,4,5}: colors 2,2,0 - non-monochromatic ✓
    
    All edges are non-monochromatic. Good.
    
    Now verify no 2-coloring exists: The Fano plane is known to be non-2-colorable.
    This is a classic result. Any 2-coloring of the 7 points of the Fano plane
    must have a monochromatic line.
    """
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
