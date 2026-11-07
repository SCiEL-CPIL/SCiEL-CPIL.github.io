"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    """
    Return a 3-chromatic 3-uniform hypergraph on 9 vertices with minimum edge count.
    The chromatic number of a 3-uniform hypergraph is the minimum k such that no
    (k-1)-coloring exists. We need chi=3, so no 2-coloring exists.
    
    Key insight: A 3-uniform hypergraph is 2-colorable iff its edges can be partitioned
    into two sets such that no edge is monochromatic in either part... wait, that's not right.
    
    Actually: A hypergraph is 2-colorable (property B) iff there exists a partition of vertices
    into two parts such that no edge is entirely contained in one part.
    
    So chi=3 means: for EVERY partition of vertices into two parts, there exists at least one
    edge that is entirely in one part (monochromatic).
    
    We want to minimize the number of edges.
    
    Known result: The minimum number of edges in a 3-uniform 3-chromatic hypergraph on n vertices.
    For n=9, we need to find the minimum.
    
    Let's think about this differently. A 3-uniform hypergraph is NOT 2-colorable iff it contains
    a sub-hypergraph that is not 2-colorable. The smallest non-2-colorable 3-uniform hypergraph
    is the Fano plane (7 vertices, 7 edges) - but that's on 7 vertices.
    
    Wait, the Fano plane has 7 vertices and 7 edges, and it's 2-colorable? No, the Fano plane
    is NOT 2-colorable. Let me verify: the Fano plane has 7 points and 7 lines (each line has 3 points).
    Can we 2-color the 7 points so no line is monochromatic? 
    
    Actually, the Fano plane IS 2-colorable... no wait. Let me think again.
    
    The property B for the Fano plane: Is there a 2-coloring with no monochromatic line?
    The Fano plane has girth 3 (in the hypergraph sense), and it's known that the Fano plane
    is NOT 2-colorable. Yes, the Fano plane is the smallest non-2-colorable 3-uniform hypergraph,
    with 7 vertices and 7 edges.
    
    But we need 9 vertices. So we can take the Fano plane on 7 vertices and add 2 more vertices
    in some way. The question is: what's the minimum number of edges for a 3-chromatic 3-uniform
    hypergraph on exactly 9 vertices?
    
    Since the Fano plane on 7 vertices already has 7 edges and is 3-chromatic, we can add 2 more
    vertices. If we just add isolated vertices, the chromatic number stays 3. But wait - do isolated
    vertices count? The problem says "distinct triples on 0..8", so vertices 0-8 must all be present.
    If a vertex is not in any edge, it's isolated. The coloring still works because isolated vertices
    can be any color.
    
    So one approach: take the Fano plane on vertices {0,1,2,3,4,5,6}, and leave vertices 7,8 isolated.
    This gives 7 edges. But is there a way to do it with fewer than 7 edges?
    
    The minimum number of edges in a non-2-colorable 3-uniform hypergraph is 7 (the Fano plane).
    So we need at least 7 edges.
    
    But wait - can we have a non-2-colorable 3-uniform hypergraph with 9 vertices and fewer than 7 edges?
    No, because any sub-hypergraph that is non-2-colorable must have at least 7 edges (Fano plane is
    minimal). So the minimum is 7.
    
    Let me construct the Fano plane on vertices 0-6:
    Fano plane edges:
    {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    
    Let me verify this is the Fano plane. The Fano plane can be constructed as follows:
    Points: 0,1,2,3,4,5,6
    Lines: 
    012, 034, 056, 135, 146, 236, 245
    
    Let me check: each point appears in exactly 3 lines:
    0: 012, 034, 056 ✓
    1: 012, 135, 146 ✓
    2: 012, 236, 245 ✓
    3: 034, 135, 236 ✓
    4: 034, 146, 245 ✓
    5: 056, 135, 245 ✓
    6: 056, 146, 236 ✓
    
    Each pair of points appears in exactly one line:
    01: 012
    02: 012
    03: 034
    04: 034
    05: 056
    06: 056
    12: 012
    13: 135
    14: 146
    15: 135
    16: 146
    23: 236
    24: 245
    25: 245
    26: 236
    34: 034
    35: 135
    36: 236
    45: 245
    46: 146
    56: 056
    All 21 pairs covered exactly once. ✓
    
    Now I need to verify the Fano plane is NOT 2-colorable.
    Assume a 2-coloring exists. WLOG color 0 = A.
    In line 012: not all same, so at least one of {1,2} is B.
    In line 034: at least one of {3,4} is B.
    In line 056: at least one of {5,6} is B.
    
    This is a well-known result. The Fano plane is not 2-colorable.
    
    Now for the coloring with 3 colors: we need a proper 3-coloring where no edge is monochromatic.
    Since the Fano plane is 3-chromatic, there exists a 3-coloring with no monochromatic edge.
    
    Let me find one. One known 3-coloring of the Fano plane:
    Color 0 = 0
    Color 1 = 1
    Color 2 = 2
    Color 3 = 1
    Color 4 = 2
    Color 5 = 0
    Color 6 = 1
    
    Let me check each edge:
    012: colors 0,1,2 - not mono ✓
    034: colors 0,1,2 - not mono ✓
    056: colors 0,0,1 - not mono ✓
    135: colors 1,1,0 - not mono ✓
    146: colors 1,2,1 - not mono ✓
    236: colors 2,1,1 - not mono ✓
    245: colors 2,2,0 - not mono ✓
    
    Great! Now vertices 7 and 8 are isolated, so they can be any color. Let's give them color 0.
    
    Final coloring: [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    Edges: [[0,1,2], [0,3,4], [0,5,6], [1,3,5], [1,4,6], [2,3,6], [2,4,5]]
    """
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
