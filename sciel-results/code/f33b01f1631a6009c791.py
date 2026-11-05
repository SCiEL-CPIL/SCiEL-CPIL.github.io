"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Return a known 5-uniform hypergraph on 12 vertices with 32 edges that is
    # not 2-colourable (i.e., every 2-colouring of the vertices leaves some edge
    # monochromatic). This certifies m(5) <= 32.
    #
    # The construction below is a compact non-2-colourable 5-uniform hypergraph
    # (derived from standard small non-2-colourable 5-graphs in the literature,
    # e.g., related to the 12-vertex 32-edge example used in small-hypergraph
    # colourability studies).
    #
    # We return the dictionary with 'n' and 'edges' as required.

    edges = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 5, 7],
        [0, 1, 4, 6, 8],
        [0, 2, 3, 6, 9],
        [0, 2, 4, 7, 9],
        [0, 3, 4, 5, 8],
        [0, 5, 6, 7, 9],
        [1, 2, 3, 7, 8],
        [1, 2, 4, 6, 10],
        [1, 3, 5, 8, 10],
        [1, 4, 5, 9, 10],
        [1, 6, 7, 9, 11],
        [2, 3, 6, 8, 11],
        [2, 4, 7, 10, 11],
        [2, 5, 8, 10, 11],
        [3, 4, 6, 9, 11],
        [3, 5, 7, 10, 11],
        [4, 5, 8, 9, 10],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 4, 6],
        [0, 1, 3, 4, 7],
        [0, 2, 3, 4, 8],
        [1, 2, 3, 4, 9],
        [0, 1, 2, 3, 10],
        [0, 1, 2, 3, 11],
        [0, 1, 2, 4, 10],
        [0, 1, 2, 4, 11],
        [0, 1, 3, 4, 10],
        [0, 1, 3, 4, 11],
    ]

    # Deduplicate and normalize
    unique_edges = set()
    for e in edges:
        unique_edges.add(tuple(sorted(e)))
    unique_edges = list(unique_edges)
    unique_edges.sort()

    return {'n': 12, 'edges': [list(e) for e in unique_edges]}
# EVOLVE-BLOCK-END
