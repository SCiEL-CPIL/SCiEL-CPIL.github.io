"""Rewritten program: hard-code the known optimal Fano-plane 3-uniform hypergraph on 9 vertices.

The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges that is 3-chromatic
(and not 2-colorable). By adding two isolated vertices (7,8) and keeping the same edge set,
we obtain a 3-uniform hypergraph on vertices 0..8 with exactly 7 edges that is still not
2-colorable (the isolated vertices can be colored arbitrarily without affecting 2-colorability
of the core). 7 edges is the theoretical minimum for a 3-chromatic 3-uniform hypergraph,
so this is optimal.

A valid 3-coloring of the Fano plane: assign colors so that every edge is non-monochromatic.
One standard 3-coloring of the Fano plane (vertices 0..6):
  The Fano plane edges are:
    (0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)
  A proper 3-coloring (each edge has all three colors or at least two colors):
  Actually, the Fano plane is not 3-colorable in the sense that every edge gets all three
  distinct colors simultaneously? Let me check: the Fano plane has chromatic number 3,
  meaning it cannot be 2-colored, but it CAN be 3-colored. In a 3-coloring, each edge
  must be non-monochromatic.

  Let me find a valid 3-coloring:
  Try: 0:0, 1:1, 2:2, 3:1, 4:2, 5:0, 6:1
  Check edges:
    (0,1,2) -> (0,1,2) OK
    (0,3,4) -> (0,1,2) OK
    (0,5,6) -> (0,0,1) -> non-monochromatic OK
    (1,3,5) -> (1,1,0) -> non-monochromatic OK
    (1,4,6) -> (1,2,1) -> non-monochromatic OK
    (2,3,6) -> (2,1,1) -> non-monochromatic OK
    (2,4,5) -> (2,2,0) -> non-monochromatic OK
  All edges are non-monochromatic. Good.
  Vertices 7,8 can be any color, say 0.
"""

def construct():
    # Fano plane edges on vertices 0..6
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    # 3-coloring: vertices 0..6 as above, vertices 7,8 = 0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
