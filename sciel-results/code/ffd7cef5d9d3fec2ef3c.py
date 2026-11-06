"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # K3,13 + triangle on the 3-vertex part
    edges = []
    # Triangle on vertices 0, 1, 2
    edges.append([0, 1])
    edges.append([1, 2])
    edges.append([0, 2])
    # Cross edges between {0, 1, 2} and {3, ..., 15}
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    return edges
# EVOLVE-BLOCK-END
