import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Pre-computed optimal solution with 10 added edges.
    # Hub 12 connected to all odd vertices except its path neighbors 11 and 13.
    # This ensures diameter <= 4 and triangle-free property (bipartite).
    added_edges = [
        [12, 1],
        [12, 3],
        [12, 5],
        [12, 7],
        [12, 9],
        [12, 15],
        [12, 17],
        [12, 19],
        [12, 21],
        [12, 23]
    ]
    return added_edges

# EVOLVE-BLOCK-END
