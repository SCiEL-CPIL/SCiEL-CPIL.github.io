"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    edges = []
    # Component 1: K_3,3 on vertices 0-5
    # Partitions: {0, 1, 2} and {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: K_3,3 on vertices 6-11
    # Partitions: {6, 7, 8} and {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
