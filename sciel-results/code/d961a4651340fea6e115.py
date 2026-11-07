"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Solution with 9 edges
    # Hubs at 11 and 22
    # Edges to 11: 0, 2, 4, 6, 8
    # Bridge: 11-22
    # Edges to 22: 15, 17, 19
    added_edges = [
        [0, 11],
        [2, 11],
        [4, 11],
        [6, 11],
        [8, 11],
        [11, 22],
        [15, 22],
        [17, 22],
        [19, 22]
    ]
    return added_edges
# EVOLVE-BLOCK-END
