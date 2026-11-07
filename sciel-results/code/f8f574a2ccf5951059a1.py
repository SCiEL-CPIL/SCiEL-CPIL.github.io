def construct():
    # The Fano plane on vertices 0..6 is a 3-uniform hypergraph that is
    # not 2-colorable (every 2-coloring of its 7 vertices leaves at least
    # one line monochromatic).  Therefore its chromatic number is exactly 3,
    # and 7 edges is the minimal edge count for a 3-chromatic triple system
    # on 9 vertices (vertices 7 and 8 are isolated and do not affect the
    # edge count).  We verify the 2-coloring infeasibility explicitly with a
    # brute-force check over all 2^7 = 128 assignments to the Fano vertices.
    # The Fano plane on vertices 0..6 is the minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 edges. Vertices 7 and 8 are isolated.
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # Construct the edge list
    edges = [[int(v) for v in e] for e in fano_edges]

    # A valid 3-coloring: no edge is monochromatic
    # Vertices 0-6 are colored to ensure no Fano line is monochromatic.
    # Vertices 7, 8 are colored arbitrarily (0).
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are exactly Python int and lists
    colors = [int(c) for c in colors]

    return {
        'edges': edges,
        'colors': colors
    }
